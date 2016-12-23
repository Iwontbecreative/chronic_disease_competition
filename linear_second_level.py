import numpy as np
import pandas as pd
from helper import cv_split, output_csv
from helper import add_stack_features
from columns import to_keep_lr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def mini_add_features(data):
    data['pourc_ald'] = data.nombre_sej_ald / data.nombre_sej
    return data

FOLDER = 'run'
FILENAME = 'linear_second_level'

clf = LinearRegression()

data = pd.read_csv('proc_data.csv', usecols=to_keep_lr)
data.index = data.id
data.drop(['id'], axis=1, inplace=True)
data = add_stack_features(data, 'train')
data = mini_add_features(data)
data.drop(['first_level/XGBOOST_'], axis=1, inplace=True)

data_wo_label = data.loc[:, data.columns != 'label']
clf.fit(data_wo_label, data.label)

to_keep_lr.remove('label')
test = pd.read_csv('proc_test.csv', usecols=to_keep_lr)
test.index = test.id
test.drop(['id'], axis=1, inplace=True)
test = add_stack_features(test, 'test')
test = mini_add_features(test)
test.drop(['first_level/XGBOOST_'], axis=1, inplace=True)

for i, j in zip(data_wo_label.columns, clf.coef_):
    print(i, j)

pred = pd.Series(clf.predict(test), index=test.index)
pred = pred.apply(lambda p: max(0, p))
pred = pred.apply(lambda p: min(1, p))
print('Average prediction is :', pred.mean())
output_csv(pred, FOLDER, FILENAME + '_test', True, True)
