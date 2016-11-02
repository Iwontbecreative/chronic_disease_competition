"""
Processing data after it has been cleaned by transform_data.py
Can be run either in LOCAL (do cross-validation) or submit mode.
XGBOOST decides whether we use XGBoost gradient boosted trees.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from helper import output_csv, cv_split, one_vs_all, one_vs_previous
from helper import add_features, add_stack_features
from columns import to_keep, to_keep_2015
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

LOCAL = True
XGBOOST = False
CV = 2
CV_METHOD = cv_split
USE_STACKED_FEATURES = False
STACK = False

FOLDER = 'first_level' if STACK else 'run'
FILENAME = 'RFR_STD_50' if STACK else 'prediction'

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
        'eta': 0.2,
    }

else:
    # We don't use GBR because XGBoost is much faster and less memory-hungry.
    # clf = GradientBoostingRegressor(n_estimators=40, max_depth=13,
                                    # max_features=7, verbose=1,
                                    # random_state=1)
    # Works best with 50.
    clf = RandomForestRegressor(n_estimators=40, random_state=1, n_jobs=4,
                                max_features=7, bootstrap=False)

start = datetime.now()
print('Started process at {:%H:%M:%S}'.format(start))


data = pd.read_csv('proc_data.csv', usecols=to_keep_2015)
data.index = data.id
data.drop(['id'], axis=1, inplace=True)
data.fillna(-1000, inplace=True)  # needed to handle pdmreg and pdmza
data = add_features(data)

if USE_STACKED_FEATURES:
    data = add_stack_features(data, 'train')

start_train = datetime.now()
print('Started training at {:%H:%M:%S}'.format(start_train))

if LOCAL:
    scores, f_i, stored_results = [], [], []
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
        # We'd rather work with pandas objects all the way.
        pred = pd.Series(pred, index=Xtest.index)
        pred = pred.apply(lambda p: max(0, p))
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
        output_csv(stacked, FOLDER, FILENAME + '_train', True, False)
        LOCAL = False


# We don't use else because STACK=True changes LOCAL to False
# so that the first_levels are generated in one pass.
if not LOCAL:
    data_wo_label = data.loc[:, data.columns != 'label']
    if XGBOOST:
        xgb_train = xgb.DMatrix(data_wo_label, data.label)
        clf = xgb.train(param, xgb_train, num_boost_round=70,
                        verbose_eval=1, evals=[(xgb_train, 'Train')])
    else:
        clf.fit(data_wo_label, data.label)
        for col, imp in zip(data_wo_label, clf.feature_importances_*100):
            print(col, ':', imp)
    print('Training took : {}'.format(datetime.now()-start_train))

    # Ugly call to Garbage collector to work around ram limits.
    del data

    # Run on test then output predictions.
    to_keep.remove('label')
    to_keep_2015.remove('label')
    test = pd.read_csv('proc_test.csv', usecols=to_keep_2015)
    test.index = test.id
    test.drop(['id'], axis=1, inplace=True)
    test = add_features(test)
    if USE_STACKED_FEATURES:
        test = add_stack_features(test, 'test')
    if XGBOOST:
        test = xgb.DMatrix(test)
    ypred = clf.predict(test)
    ypred = pd.Series(ypred, index=test.index)
    ypred = ypred.apply(lambda p: max(0, p))
    print('Average prediction is :', sum(ypred)/len(ypred))
    output_csv(ypred, FOLDER, FILENAME + '_test', True, False)

print('Total process took : {}'.format(datetime.now()-start))
