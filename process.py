import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
import csv

LOCAL = False

def output_csv(pred):
    with open('run/output_{}.csv'.format(datetime.now()), 'w') as output:
        writer = csv.writer(output, delimiter=';')
        writer.writerow(['id', 'cible'])
        writer.writerows(enumerate(pred))

def split(data, before=2013):
    """
    Small utility to split data based on years
    """
    mask = data.an < before
    Xtrain, Xtest = data[mask], data[~mask]
    ytrain, ytest = Xtrain.label, Xtest.label
    Xtrain.drop(['label'], axis=1, inplace=True)
    Xtest.drop(['label'], axis=1, inplace=True)
    return Xtrain, Xtest, ytrain, ytest

def add_features(data):
    data['pourc_ald'] = data.nombre_sej_ald / data.nombre_sej
    return data

start = datetime.now()

print('Started process at {:%H:%M:%S}'.format(start))
data = pd.read_csv('proc_data.csv')
data = add_features(data)
RFR = RandomForestRegressor(n_estimators=30, random_state=1, n_jobs=4)

if LOCAL:
    Xtrain, Xtest, ytrain, ytest = split(data)
    start_train = datetime.now()
    print('Started training at {:%H:%M:%S}'.format(start_train))
    RFR.fit(Xtrain, ytrain)
    print('Training took : {} seconds'.format(datetime.now()-start_train))
    pred = RFR.predict(Xtest)
    print('Score was :', mean_squared_error(ytest, pred)**0.5*100)
    print('Feature importances were :')
    for col, imp in zip(Xtrain.columns, RFR.feature_importances_*100):
        print(col, ':', imp)

else:
    labels = data.label
    data.drop(['label'], axis=1, inplace=True)
    start_train = datetime.now()
    print('Started training at {:%H:%M:%S}'.format(start_train))
    RFR.fit(data, labels)
    print('Training took : {} seconds'.format(datetime.now()-start_train))

    del data
    test = pd.read_csv('proc_test.csv')
    test = add_features(test)
    ypred = RFR.predict(test)
    print('Average prediction is :', sum(ypred)/len(ypred))
    output_csv(ypred)

print('Total process took : {}'.format(datetime.now()-start))
