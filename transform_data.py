"""
Transform data so it is easier on ram.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import glob

sources = ['data2.csv', 'test2.csv']

def pre_treatment():
    """
    We want to use the 'raison_sociale' field on both files.
    We extract a labelencoder to be able to use this information.
    """
    names = []
    for source in sources:
        ser = pd.read_csv(source, usecols=['Raison sociale'], sep=';').ix[:, 0]
        names.extend(list(ser))
    le = LabelEncoder()
    le.fit(names)
    return le

def generate_hospidiag(cols, exceptions):
    """
    We want to be able to handle hospidiag data so it can be merged nicely
    with the rest of the information we have.
    Cols describes which columns (text) we should keep.
    Exceptions are columns that are not handled as numbers.
    """
    print('Working on hospidiag')
    merged_csv = pd.DataFrame()
    for csv in glob.glob('hospidiag/*.csv'):
        year = int(csv[-8:-4])
        # Due to a bug in pandas we load all the columns then
        # select the ones we keep
        current = pd.read_csv(csv)[cols]
        current = current.rename(columns = {'finess':'eta'})
        # We add the year as a variable so that our merge
        # works on both eta and year.
        current['an'] = [year] * len(current)
        merged_csv = pd.concat([merged_csv, current])
    # Reindex to avoid any weird interaction with multiple indexes.
    merged_csv.index = list(range(len(merged_csv)))
    # For all columns aside from eta, we want to transform it to numeric
    # despite many variations of weird formatting
    for col in merged_csv.columns:
        if (col not in ('an', 'eta')) and (col not in exceptions):
            merged_csv[col] = merged_csv[col].apply(handle_ugliness)
    le = LabelEncoder()
    for col in exceptions:
        merged_csv[col] = merged_csv[col].fillna('Void')
        merged_csv[col] = le.fit_transform(merged_csv[col])
    print('Finished working on hospidiag')
    return merged_csv

def handle_ugliness(x):
    """
    We want to provide a nice way to convert a column to float.
    This accounts for NaN, numeric values, and annoying , instead of .
    """
    if isinstance(x, str):
        try:
            return float(x.replace(',', '.'))
        except ValueError:
            return -1
    else:
        if pd.isnull(x):
            return -1000 # Need to be negative for e.g margin rate.
        return x

cols = ['finess']
# Add a bunch of numerical columns...
cols.extend(['A{}'.format(i) for i in range(7, 16)])
cols.extend(['F{}_O'.format(i) for i in range(1, 13)])
cols.extend(['F{}_D'.format(i) for i in range(1, 13)])
cols.extend(['P{}'.format(i) for i in range(1, 17)])
cols.extend(['RH{}'.format(i) for i in range(1, 11)])
cols.remove('RH7') # Lacking in 2008, too much trouble.
cols.extend(['CI_AC{}'.format(i) for i in range(1, 10)])
cols.extend(['CI_A{}'.format(i) for i in range(1, 16)])
cols.extend(['CI_E{}'.format(i) for i in range(1, 8)])
cols.extend(['CI_F{}_O'.format(i) for i in range(1, 18)])
cols.extend(['CI_F{}_D'.format(i) for i in range(1, 18)])
cols.extend(['CI_RH{}'.format(i) for i in range(1, 12)])
# The categorical ones.
exceptions = ['champ_pmsi', 'cat', 'taille_MCO', 'taille_M', 'taille_C',
              'taille_O']
cols.extend(exceptions)
hospidiag = generate_hospidiag(cols, exceptions)
le = pre_treatment()


for source in sources:
    print('Starting to process :', source)
    data = pd.read_csv(source, sep=';', dtype={'Finess': str})

    is_test = 'test' in source

    if is_test:
        data.drop(['id'], axis=1, inplace=True)

    columns = ['eta', 'nom_eta', 'prov_patient', 'dom_acti', 'age',
               'nombre_sej_ald', 'nombre_sej', 'an']

    if not is_test:
        columns.append('label')

    data.columns = columns

    # We want to manipulate before treating so changes can be done on both datasets (e.g for eta).
    data = pd.merge(data, hospidiag, how='left', on=['eta', 'an'])

    # Just manipulate for treatment
    data.eta = data.eta.apply(lambda s: int(str(s).replace('2A', '2000').replace('2B', '2001')))
    data.dom_acti = data.dom_acti.apply(lambda s: int(s[1:3]))
    data.prov_patient = data.prov_patient.apply(lambda s: s.replace('Inconnu', '0').replace('2A', '2000').replace('2B', '2001'))
    data.prov_patient = data.prov_patient.apply(lambda s: int(s.split('-')[0]))
    data.age = data.age.apply(lambda s: int(s == '>75 ans'))
    data.nom_eta = le.transform(data.nom_eta)

    # Export to csv
    suffix = 'test' if is_test else 'data'
    data.to_csv('proc_{}.csv'.format(suffix), index=False)
