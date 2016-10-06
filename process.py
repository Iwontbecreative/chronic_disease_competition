"""
Processing data after it has been cleaned by transform_data.py
Can be run either in LOCAL (do cross-validation) or submit mode.
XGBOOST decides whether we use XGBoost gradient boosted trees.
"""

from datetime import datetime
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from helper import output_csv, cv_split, one_vs_all, one_vs_previous
# We use XGBoost's version of Gradient Boosting over SkLearn's one because of
# the approximate tree building algorithm which is much faster with nearly no
# performance loss.
# See Section 3 of paper https://arxiv.org/pdf/1603.02754v3.pdf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

LOCAL = True
XGBOOST = False
CV = 1
CV_METHOD = cv_split
USE_STACKED_FEATURES = True
STACK = False

FOLDER = 'first_level' if STACK else 'run'
FILENAME = 'XGBOOST_STD' if STACK else 'prediction'

average_year = {
        2008: 0.1027,
        2009: 0.1129,
        2010: 0.1258,
        2011: 0.1317,
        2012: 0.1384,
        2013: 0.1483,
        2014: 0.161,
        2015: 0.16,
        }



def add_features(data):
    """
    Feature engineering other than cleaning data goes here.
    """
    data['pourc_ald'] = data.nombre_sej_ald / data.nombre_sej
    data['diff_ald'] = data.nombre_sej - data.nombre_sej_ald
    if 'CI_AC1' in data.columns and 'CI_AC4' in data.columns:
        fillme = []
        # Workaround to avoid division by zero.
        for beds, rea_beds in zip(data.CI_AC1, data.CI_AC4):
            fillme.append(rea_beds / beds if beds else 0)
        data['part_lits_rea'] = fillme
    return data

def add_stack_features(data, kind):
    """
    Add stacked features to dataset `kind` € {'train', 'test'}
    """
    # We need to add the features in the right order.
# FIXME: Code is ugly
    features = [f for f in glob.glob('first_level/*.csv')]
    features.sort()
    for col in features:
        if kind in col:
            data[col.split(kind)[0]] = pd.read_csv(col, sep=';').cible
    return data

start = datetime.now()
print('Started process at {:%H:%M:%S}'.format(start))

# Columns to keep. Note that keeping 40+ columns will usually swap (on 4Gb ram)
# using Random Forest Regressors or Gradient Boosted Trees.
to_keep = ['eta', 'nom_eta', 'prov_patient', 'dom_acti', 'age',
           'nombre_sej_ald', 'nombre_sej', 'an', 'label']
# Extend with those that passed the treshold (> 0.1% contrib)
to_keep.extend(['A9', 'A12', 'A13', 'A14', 'CI_AC1', 'CI_AC4',
                'CI_AC6', 'CI_AC7', 'CI_RH4', 'CI_RH1', 'CI_RH3',
                'CI_A5', 'CI_A12', 'CI_A15', 'P2', 'P13', 'P9',
                'P12', 'P14', 'P15', 'A1bis', 'A2bis', 'A4bis',
                'A5bis', 'cat'])
to_keep.extend(['RH{}'.format(i) for i in range(2, 6)])

to_keep_2015 = ['eta', 'nom_eta', 'prov_patient', 'dom_acti', 'age',
                'nombre_sej_ald', 'nombre_sej', 'an', 'label']
to_keep_2015.extend(['A1bis', 'A2bis', 'A4bis', 'A5bis', 'cat',
                     'A9', 'A12', 'CI_A5', 'CI_A12', 'CI_A15', 'P2',
                     'P13', 'P12', 'P15', 'CI_A16_6'])

# data = pd.read_csv('proc_data.csv', usecols=to_keep)
data = pd.read_csv('proc_data.csv', usecols=to_keep_2015)
data.fillna(-1000, inplace=True)  # needed to handle pdmreg and pdmza
data = add_features(data)

if USE_STACKED_FEATURES:
    data = add_stack_features(data, 'train')


### ML Tuning ###

if XGBOOST:
    param = {
        'booster': 'gbtree',
        'max_depth': 20,
        'objective': 'reg:linear',
        'silent': 1,
        'nthread': 4,
        'eval_metric': 'rmse',
        'seed': 1,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'colsample_bylevel': 0.9,
        # 'base_score': 0.14, Comment out since scores depend too much on
        'eta': 0.2,
    }
else:
    # We don't use GBR because XGBoost is much faster and less memory-hungry.
    # clf = GradientBoostingRegressor(n_estimators=40, max_depth=13,
                                    # max_features=7, verbose=1,
                                    # random_state=1)
    # Works best with 50.
    clf = RandomForestRegressor(n_estimators=50, random_state=1, n_jobs=4,
                                max_features=7, bootstrap=False)

if LOCAL:
    start_train = datetime.now()
    print('Started training at {:%H:%M:%S}'.format(start_train))
    scores = []
    f_i = []
    stored_results = []
    for i in range(CV):
        print('CV round ', i)
        if i:
            del Xtrain, Xtest
        Xtrain, Xtest, ytrain, ytest = CV_METHOD(data, i)
        if XGBOOST:
            xgb_train = xgb.DMatrix(Xtrain, ytrain)
            xgb_test = xgb.DMatrix(Xtest, ytest)
            eval = [(xgb_train, 'Train'), (xgb_test, 'Test')]
            clf = xgb.train(param, xgb_train, num_boost_round=80,
                            evals=eval, verbose_eval=1,
                            early_stopping_rounds=3)
            pred = clf.predict(xgb_test)
        else:
            clf.fit(Xtrain, ytrain)
            pred = clf.predict(Xtest)
            f_i.append(clf.feature_importances_*100)
        # With GradientBoostedTrees, scores can be had even though y € [0, inf[
        # Correct this by setting negative scores to 0.
        pred = [max(0, p) for p in pred]
        # We need to save the scores with the weights.
        if STACK:
            stored_results.append(pd.Series(pred, ytest.index))
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

    if STACK:
        stacked = pd.concat(stored_results)
        output_csv(stacked, FOLDER, FILENAME + '_train', False)
        LOCAL = False


# We don't use else because STACK=True changes LOCAL to False
# so that the first_levels are generated in one pass.
if not LOCAL:
    start_train = datetime.now()
    print('Started training at {:%H:%M:%S}'.format(start_train))
    data_wo_label = data.loc[:, data.columns != 'label']
    if XGBOOST:
        xgb_train = xgb.DMatrix(data_wo_label, data.label)
        clf = xgb.train(param, xgb_train, num_boost_round=70,
                        verbose_eval=1, evals=[(xgb_train, 'Train')])
    else:
        clf.fit(data_wo_label, data.label)
        for col, imp in zip(data_wo_label, clf.feature_importances_*100):
            print(col, ':', imp)
    print('Training took : {} seconds'.format(datetime.now()-start_train))

    # Ugly call to Garbage collector to work around ram limits.
    del data

    # Run on test then output predictions.
    to_keep.remove('label')
    to_keep_2015.remove('label')
    test = pd.read_csv('proc_test.csv', usecols=to_keep_2015)
    test = add_features(test)
    if USE_STACKED_FEATURES:
        test = add_stack_features(test, 'test')
    if XGBOOST:
        test = xgb.DMatrix(test)
    ypred = clf.predict(test)
    ypred = [max(0, p) for p, z in ypred]
    print('Average prediction is :', sum(ypred)/len(ypred))
    output_csv(ypred, FOLDER, FILENAME + '_test', False)

print('Total process took : {}'.format(datetime.now()-start))
