import glob
import pandas as pd
import numpy as np

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

### Splitting functions, used for cross_val ###

# FIXME: Once the cross-validation methods will be settled, use
# sklearn's StratifiedKFold instead. For now those 3 functions are ugly.


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


def one_vs_previous(data, i, previous_years=2):
    """
    Predict one year according to the 'previous_years' years before.
    For this function we must have CV <= 5-previous_years
    """
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


def add_features(data):
    """
    Feature engineering other than cleaning data goes here.
    """
    dept_code = data.eta.apply(retrieve_dept_code)
    data['dept_code'] = dept_code
    data['prov_egal_lieu'] = [int(i) for i in data.prov_patient == dept_code]
    # log transforms.
    data.nombre_sej_ald = np.log1p(data.nombre_sej_ald)
    data.nombre_sej = np.log1p(data.nombre_sej)
    data['pourc_ald'] = data.nombre_sej_ald / data.nombre_sej
    data['diff_ald'] = data.nombre_sej - data.nombre_sej_ald

    # naive bayes
    mean_val, benchmark = {}, {}
    for nom_eta in data.nom_eta.unique():
        masked = data[data.nom_eta == nom_eta]
        mean_val[nom_eta] = masked.label.mean()
        ln2 = np.log(2)
        benchmark[nom_eta] = masked[(masked.nombre_sej == ln2) & (masked.nombre_sej_ald == ln2)].label.mean()
    data['bayes_nom_eta'] = data.nom_eta.apply(lambda e: mean_val[e])
    data.bayes_nom_eta.fillna(-1, inplace=True)
    data['bayes_benchmark'] = data.nom_eta.apply(lambda e: benchmark[e])
    data.bayes_benchmark.fillna(-1, inplace=True)

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
            data[col.split(kind)[0]] = pd.read_csv(col, sep=';').cible
    return data
