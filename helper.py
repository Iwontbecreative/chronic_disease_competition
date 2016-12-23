import glob
import pandas as pd
import numpy as np
from datetime import datetime


def output_csv(pred, folder, filename, test=True, time=True):
    # Depending on the length we want different zero padding
    magic = 664524 if test else 1879841
    predictions = pd.Series(0, np.arange(magic))
    predictions = pd.concat([predictions, pred], axis=1).drop(0, axis=1)
    predictions.fillna(0, inplace=True)
    predictions.index.name, predictions.columns = 'id', ['cible']
    time_at = datetime.now() if time else ""
    predictions.to_csv('{folder}/{file}_{time}.csv'.format(folder=folder,
                                                           file=filename,
                                                           time=time_at),
                                                           sep=';',
                                                           header=True)


def stack_csv(pred, folder, filename, test=True, time=True):
    time_at = datetime.now() if time else ""
    pred.index.name, pred.name = 'id', 'cible'
    pred.to_csv('{folder}/{file}_{time}.csv'.format(folder=folder,
                                                           file=filename,
                                                           time=time_at),
                                                           sep=';',
                                                           header=True)

# Splitting functions, used for cross_val


def split(data, before=2012):
    """
    Small utility to split data based on years
    """
    data = data[data.an < 2013]
    mask = data.an < before
    Xtrain, Xtest = data[mask], data[~mask]
    ytrain, ytest = Xtrain.label, Xtest.label
    mask = ~Xtrain.columns.isin(['label'])
    return Xtrain.loc[:, mask], Xtest.loc[:, mask], ytrain, ytest


def cv_split(data, i):
    """
    Returns 3 years of train, 2 years of test.
    1) Train < 2012, Test = 2012-2013
    2) Train > 2009, Test = 2008-2009
    For this function we must have CV <= 2
    """
    if not i:
        return split(data)
    else:
        a, b, c, d = split(data, before=2010)
        return b, a, d, c


def one_vs_previous(data, i, previous_years=1):
    """
    Predict one year according to the 'previous_years' years before.
    For this function we must have CV <= 5-previous_years
    """
    if not i:
        Xtrain = data[data.an == 2009]
        Xtest = data[data.an == 2008]
    else:
        year_to_test = 2008 + i + previous_years
        year = data.an == year_to_test
        previous = data.an.isin(np.arange(2008 + i, year_to_test))
        Xtrain, Xtest = data[previous], data[year]
    ytrain, ytest = Xtrain.label, Xtest.label
    mask = ~Xtrain.columns.isin(['label'])
    return Xtrain.loc[:, mask], Xtest.loc[:, mask], ytrain, ytest


def one_vs_all(data, i):
    mask = data.an != 2008 + i
    Xtrain, Xtest = data[mask], data[~mask]
    ytrain, ytest = Xtrain.label, Xtest.label
    mask = ~Xtrain.columns.isin(['label'])
    return Xtrain.loc[:, mask], Xtest.loc[:, mask], ytrain, ytest


def naive_bayes(data):
    data = data[data.an < 2012]
    mean_val, eta_benchmark = {}, {}
    global_bench = benchmark(data)
    for nom_eta in data.nom_eta.unique():
        masked = data[data.nom_eta == nom_eta]
        mean_val[nom_eta] = masked.label.mean()
        result_benchmark = compare_to_benchmark(masked, global_bench)
        l = len(result_benchmark)
        eta_benchmark[nom_eta] = sum(result_benchmark)/l
    return mean_val, eta_benchmark


def benchmark(data):
    benchmark = {}
    for i in range(1, 7):
        for j in range(i, 7):
            mask = (data.nombre_sej_ald == i) & (data.nombre_sej == j)
            benchmark['{}{}'.format(i, j)] = data[mask].label.mean()
    return benchmark


def compare_to_benchmark(eta, global_bench):
    scores = []
    for i in range(1, 7):
        for j in range(i, 7):
            mask = (eta.nombre_sej_ald == i) & (eta.nombre_sej == j)
            val = eta[mask].label.mean()
            if val:
                scores.append(val - global_bench['{}{}'.format(i, j)])
    if not scores:
        scores.append(-1000)
    return scores


# def add_features(data, mean_val, benchmark):
def add_features(data):
    """
    Feature engineering other than cleaning data goes here.
    """
    dept_code = data.eta.apply(retrieve_dept_code)
    data['dept_code'] = dept_code
    data['prov_egal_lieu'] = [int(i) for i in data.prov_patient == dept_code]

    # log transforms.
    # data['real_nombre_sej_ald'] = data.nombre_sej_ald
    # data['real_nombre_sej'] = data.nombre_sej
    data.nombre_sej_ald = np.log1p(data.nombre_sej_ald)
    data.nombre_sej = np.log1p(data.nombre_sej)
    data['pourc_ald'] = data.nombre_sej_ald / data.nombre_sej
    data['diff_ald'] = data.nombre_sej - data.nombre_sej_ald

    # Indicators of positiveness.
    # data['I_A1bis'] = data.A1bis.apply(lambda i: i>0)
    # data['I_A2bis'] = data.A2bis.apply(lambda i: i>0)
    # data['I_A4bis'] = data.A4bis.apply(lambda i: i>0)
    # data['I_A5bis'] = data.A5bis.apply(lambda i: i>0)
    # data['I_A12'] = data.A12.apply(lambda i: i>0)

    # naive bayes
    # data['bayes_nom_eta'] = data.nom_eta.apply(lambda e: mean_val.get(e, -1))
    # data['bayes_benchmark'] = data.nom_eta.apply(lambda e: benchmark.get(e, -1))
    # data['bayes_benchmark'].fillna(-1, inplace=True)
    return data


def retrieve_dept_code(s):
    return s // 10**7


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
            content = pd.read_csv(col, sep=';').cible.copy()
            data[col.split(kind)[0]+'std'] = list(content)
            content.sort_values(inplace=True)
            content.iloc[:] = np.arange(len(content))/len(content)
            content = content.sort_index()
            data[col.split(kind)[0]] = list(content)
    return data
