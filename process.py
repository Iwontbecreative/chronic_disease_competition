import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
import csv

LOCAL = True
CV = 2  # Number of folds to do on CV.

def output_csv(pred):
    with open('run/output_{}.csv'.format(datetime.now()), 'w') as output:
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
    mask = Xtrain.columns != 'label'
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

def add_features(data):
    data['pourc_ald'] = data.nombre_sej_ald / data.nombre_sej
    return data

start = datetime.now()

print('Started process at {:%H:%M:%S}'.format(start))

# Experiments...
to_keep = ['eta', 'nom_eta', 'prov_patient', 'dom_acti', 'age',
        'nombre_sej_ald', 'nombre_sej', 'an', 'label']
# Extend with those that passed the treshold (> 0.1% contrib)
to_keep.extend(['A9', 'A12', 'A13', 'A14', 'CI_AC1', 'CI_AC4',
    'CI_AC6', 'CI_AC7', 'CI_A5', 'CI_A12', 'CI_A15'])
# Add some testing ones
to_keep.extend(['cat'])

data = pd.read_csv('proc_data.csv', usecols=to_keep)
data = add_features(data)

RFR = RandomForestRegressor(n_estimators=30, random_state=1, n_jobs=4)

if LOCAL:
    start_train = datetime.now()
    print('Started training at {:%H:%M:%S}'.format(start_train))
    scores = []
    f_i = []
    for i in range(CV):
        print('CV round ', i)
        Xtrain, Xtest, ytrain, ytest = cv_split(data, i)
        RFR.fit(Xtrain, ytrain)
        pred = RFR.predict(Xtest)
        scores.append(mean_squared_error(ytest, pred)**0.5*100)
        f_i.append(RFR.feature_importances_*100)

    print('Training took : {} seconds'.format(datetime.now()-start_train))
    print('Score : Mean {}, Max {}, Min {}'.format(sum(scores)/len(scores),
                                                   max(scores), min(scores)))

    print('Mean feature importances were :')
    for col, imp in zip(Xtrain.columns, sum(f_i)/len(f_i)):
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
