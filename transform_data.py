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
    We extract:
    1) First words which occur often (top 20) as they signal the kind
       of health institution.
    2) A labelencoder to be able to use this information.
    """
    names = []
    for source in sources:
        ser = pd.read_csv(source, usecols=['Raison sociale'], sep=';').ix[:, 0]
        names.extend(list(ser))
    # 1)
    first_words = pd.Series(s.split(' ')[0] for s in names).value_counts().head(20)
    common = list(first_words.index)
    # 2)
    le = LabelEncoder()
    le.fit(names)
    return le, common

def generate_hospidiag(cols=['finess']):
    """
    We want to be able to handle hospidiag data so it can be merged nicely
    with the rest of the information we have.
    Cols describes which columns (text) we should keep.
    """

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
    return merged_csv

hospidiag = generate_hospidiag(['finess', 'A7'])
le, common = pre_treatment()


for source in sources:
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

    # Before using LE, we want to be able to retrieve the first-word.
    first_word = data.nom_eta.apply(lambda s: s.split(' ')[0])
    for word in common:
        data['prem_mot_{}'.format(word)] = first_word.apply(lambda s: int(s == word))
    data.nom_eta = le.transform(data.nom_eta)

    # Export to csv
    suffix = 'test' if is_test else 'data'
    # data.to_csv('proc_{}.csv'.format(suffix), index=False)
