import pandas as pd

def output_csv(pred, folder, filename, time=True):
    pred = pd.Series(pred)
    pred.index.name, pred.name = 'id', 'cible'
    time_at = datetime.now() if time else ""
    pred.to_csv('{folder}/{file}_{time}.csv'.format(folder=folder,
                                                    file=filename,
                                                    time=time_at), sep=';',
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
