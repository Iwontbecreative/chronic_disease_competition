import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
import csv

LOCAL = True
XGBOOST = True
CV = 4  # Number of folds to do on CV.

def output_csv(pred):
    with open('run/partial_{}.csv'.format(datetime.now()), 'w') as output:
        writer = csv.writer(output, delimiter=';')
        writer.writerow(['id', 'cible'])
        writer.writerows(enumerate(pred))

def split(data, before=2012):
    """
    Small utility to split data based on years
    """
    mask = data.an < before
    Xtrain, Xtest = data[mask], data[~mask]
    ytrain, ytest = Xtrain.label, Xtest.label
    mask = ~Xtrain.columns.isin(['label'])
    return Xtrain.loc[:, mask], Xtest.loc[:, mask], ytrain, ytest

def cv_split(data, i):
    """
    Small wrapper around split to have it work nicely in CV.
    1) Train < 2012, Test = 2012-2013
    2) Train > 2009, Test = 2008-2009
    ... #TODO
    """
    if not i:
        return split(data)
    elif i == 1:
        a, b, c, d = split(data, before=2010)
        return b, a, d, c
    return split(data)

def one_vs_previous(data, i):
    """
    Split data so that we only use the one before to predict.
    """
    year = data.an == i
    previous = data.an.isin([i-1, i-2])
    Xtrain, Xtest = data[previous], data[year]
    ytrain, ytest = Xtrain.label, Xtest.label
    mask = ~Xtrain.columns.isin(['label'])
    return Xtrain.loc[:, mask], Xtest.loc[:, mask], ytrain, ytest


def add_features(data):
    data['pourc_ald'] = data.nombre_sej_ald / data.nombre_sej
    data['diff_ald'] = data.nombre_sej - data.nombre_sej_ald
    if 'CI_AC1' in data.columns and 'CI_AC4' in data.columns:
        fillme = []
        for beds, rea_beds in zip(data.CI_AC1, data.CI_AC4):
            fillme.append(rea_beds / beds if beds else 0)
        data['part_lits_rea'] = fillme
    return data

start = datetime.now()

print('Started process at {:%H:%M:%S}'.format(start))

# Experiments...
to_keep = ['eta', 'nom_eta', 'prov_patient', 'dom_acti', 'age',
        'nombre_sej_ald', 'nombre_sej', 'an', 'label']
# Extend with those that passed the treshold (> 0.1% contrib)
to_keep.extend(['A9', 'A12', 'A13', 'A14', 'CI_AC1', 'CI_AC4',
    'CI_AC6', 'CI_AC7', 'CI_RH4', 'CI_RH1', 'CI_RH3',
    'CI_A5', 'CI_A12', 'CI_A15',  'P2', 'P13', 'P9',
    'P12', 'P14', 'P15', 'A1bis', 'A2bis', 'A4bis',
    'A5bis', 'cat'])
to_keep.extend(['RH{}'.format(i) for i in range(2, 6)])
data = pd.read_csv('proc_data.csv', usecols=to_keep)
data = data.fillna(-1000)  # needed to handle pdmreg and pdmza
data = add_features(data)

if XGBOOST:
    param = {
            'booster': 'gbtree',
            'max_depth': 20,
            'objective': 'reg:linear',
            'silent': 1,
            'nthread': 4,
            'eval_metric': 'rmse',
            'seed': 1,
            'subsample': 0.9,
            'colsample_bytree': 0.85,
            'colsample_bylevel': 0.9,
            'base_score': 0.14,
            'eta': 0.2,
    }
else:
    clf = RandomForestRegressor(n_estimators=30, random_state=1, n_jobs=4, max_features='sqrt')
    # clf = GradientBoostingRegressor(n_estimators=30, max_depth=12, max_features='sqrt',
                                    # verbose=1, random_state=1)

if LOCAL:
    start_train = datetime.now()
    print('Started training at {:%H:%M:%S}'.format(start_train))
    scores = []
    f_i = []
    for i in range(2010, CV+2009):
        print('CV round ', i)
        # Xtrain, Xtest, ytrain, ytest = cv_split(data, i)
        Xtrain, Xtest, ytrain, ytest = one_vs_previous(data, i)
        if XGBOOST:
            xgb_train = xgb.DMatrix(Xtrain, ytrain)
            xgb_test = xgb.DMatrix(Xtest, ytest)
            eval = [(xgb_train, 'Train'), (xgb_test, 'Test')]
            clf = xgb.train(param, xgb_train, num_boost_round=80,
                    evals=eval, verbose_eval=1, early_stopping_rounds=3)
            pred = clf.predict(xgb_test)
        else:
            clf.fit(Xtrain, ytrain)
            pred = clf.predict(Xtest)
            f_i.append(clf.feature_importances_*100)
        # With some algos, lower than 0 can be had. Correcting for it in all
        # case. This does not improve much but every bit counts.
        pred = [max(0, p) for p in pred]
        scores.append(mean_squared_error(ytest, pred)**0.5*100)

    print('Training took : {} seconds'.format(datetime.now()-start_train))
    print('Score : Mean {}, Max {}, Min {}'.format(sum(scores)/len(scores),
                                                   max(scores), min(scores)))
    for i, j in enumerate(scores):
        print('Score {} : {}'.format(i, j))

    if not XGBOOST:
        print('Mean feature importances were :')
        for col, imp in zip(Xtrain.columns, sum(f_i)/len(f_i)):
            print(col, ':', imp)

else:
    start_train = datetime.now()
    print('Started training at {:%H:%M:%S}'.format(start_train))
    data_wo_label = data.loc[:, data.columns != 'label']
    if XGBOOST:
        xgb_train = xgb.DMatrix(data_wo_label, data.label)
        clf = xgb.train(param, xgb_train, num_boost_round=70)
    else:
        clf.fit(data_wo_label, data.label)
        for col, imp in zip(data_wo_label, clf.feature_importances_*100):
            print(col, ':', imp)
    print('Training took : {} seconds'.format(datetime.now()-start_train))

    del data
    to_keep.remove('label')
    test = pd.read_csv('proc_test.csv', usecols=to_keep)
    test = test[test.an == 2014]
    test = add_features(test)
    if XGBOOST:
        test = xgb.DMatrix(test)
    ypred = clf.predict(test)
    ypred = [max(p, 0) for p in ypred]
    print('Average prediction is :', sum(ypred)/len(ypred))
    output_csv(ypred)

print('Total process took : {}'.format(datetime.now()-start))
