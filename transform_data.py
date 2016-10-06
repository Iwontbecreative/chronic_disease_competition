"""
Transform chronic disease data so it is easier on ram and easier to
manipulate. We also join external sources (hospidiag reports) as additional
features for our predictive model.
"""

import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

sources = ['data2.csv', 'test2.csv']


def pre_treatment():
    """
    We want to use the 'raison_sociale' field on both files to ensure classes
    are consistent across both files. One-hot-encoding does not work here
    because #{establishment} ~= 1300.
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
    We want to be able to handle hospidiag data so it can be merged nicely as
    additional features.
    This data comes from 24 csv (3 types * 8 years of data)
        hospidiag = Medical indicators on health establishments
        pdmza = Market share of establishments in their activity zone
        pdmreg = Market share of establishments in their region
    For each file, we provide cols and exceptions:
        Cols describes which columns (text) we should keep.
        Exceptions are columns that are not handled as numbers.
    """
    print('Working on merger')
    global_csv = pd.DataFrame()
    for i, kind_of in enumerate(('hospidiag', 'pdmreg', 'pdmza')):
        print('Started working on {}'.format(kind_of))
        merged_csv = pd.DataFrame()

        for csv in glob.glob('hospidiag/{}*.csv'.format(kind_of)):
            year = int(csv[-8:-4])
            current = pd.read_csv(csv, usecols=cols[i])
            current = current.rename(columns={'finess': 'eta'})
            # We add the year as a variable so that our merge
            # works on both eta and year.
            current['an'] = [year] * len(current)
            merged_csv = pd.concat([merged_csv, current])

        # Delete pdmza duplicates which will be troublesome when joining with
        # pandas (duplicates create additional rows even if joined as
        # additional columns)
        if kind_of == 'pdmza':
            merged_csv = merged_csv.drop_duplicates(['eta', 'an'])
        # Reindex to avoid issues when joining
        merged_csv.index = np.arange(len(merged_csv))
        # For all columns aside from exceptions, we want to transform them
        # to numeric despite many variations of weird formatting
        for col in merged_csv.columns:
            if (col not in ('an', 'eta')) and (col not in exceptions[i]):
                merged_csv[col] = merged_csv[col].apply(handle_ugliness)
        # For the exceptions, we LabelEncode as {#classes} is too big to
        # one-hot-encode them
        le = LabelEncoder()
        for col in exceptions[i]:
            merged_csv[col] = merged_csv[col].fillna('Void')
            if 'CI_A16' in col:
                merged_csv[col] = merged_csv[col].apply(remove_text)
            else:
                merged_csv[col] = le.fit_transform(merged_csv[col])
        # Variable 'zone' exists in both
        if 'zone' in merged_csv.columns:
            merged_csv = merged_csv.rename(columns={'zone': 'zone_{}'.format(kind_of)})
        print('Finished working on {}'.format(kind_of))

        # Handle merge logic.
        if not i:
            global_csv = merged_csv
        else:
            global_csv = pd.merge(global_csv, merged_csv, how='left', on=['eta', 'an'])

        print("Shape after merge {} : {}".format(i, global_csv.shape))
    print('Finished working on merger')
    return global_csv


def handle_ugliness(x):
    """
    Handles data format ugliness for numerical columns in data.
    This accounts for NaN, numeric values, and annoying ',' instead of '.'
    """
    if isinstance(x, str):
        try:
            return float(x.replace(',', '.'))
        except ValueError:
            return -1000
    else:
        if pd.isnull(x):
            return -1000  # Outside of the range of each feature.
        return x

def remove_text(x):
    """
    Remove all text that occurs in CI_A16 and keep only the numbers.
    """
    x = "".join(d for d in x if d.isdigit())
    if x:
        return int(x)
    else:
        return -1000

cols, exceptions = [], []


############################## COLUMN SELECTION ###########################
###########################################################################
####### Since we would rather not load everything in ram, only ############
####### select columns that proved effective.                  ############
####### Complete list of features is available on the webpage. ############
####### Adding features here will make them available for      ############
####### modelling.                                             ############
###########################################################################

##### HOSPIDIAG #####
cols_hospi = ['finess']
cols_hospi.extend(['A9', 'A12', 'A13', 'A14'])
cols_hospi.extend(['P2', 'P9', 'P12', 'P13', 'P14', 'P15'])
cols_hospi.extend(['RH{}'.format(i) for i in (2, 3, 4, 5)])
cols_hospi.extend(['CI_AC{}'.format(i) for i in (1, 4, 6, 7)])
cols_hospi.extend(['CI_A{}'.format(i) for i in (2, 5, 7, 8, 12, 15)])
cols_hospi.extend(['CI_E{}'.format(i) for i in range(1, 8)])
cols_hospi.extend(['CI_DF{}'.format(i) for i in range(1, 6)])
cols_hospi.extend(['Q{}'.format(i) for i in range(1, 12)])
cols_hospi.extend(['CI_E4_V2', 'CI_E7_V2'])
cols_hospi.extend(['CI_RH{}'.format(i) for i in range(1, 5)])
# The categorical ones.
exceptions_hospi = ['cat', 'taille_MCO', 'taille_C']
exceptions_hospi.extend(['CI_A16_{}'.format(i) for i in range(1, 7)])
cols_hospi.extend(exceptions_hospi)

cols.append(cols_hospi)
exceptions.append(exceptions_hospi)

##### PDMREG #####
cols_pdmreg = ['finess', 'zone']
cols_pdmreg.extend(['A{}bis'.format(i) for i in range(1, 6)])
cols.append(cols_pdmreg)
exceptions.append([])

##### PDMZA #####
cols_pdmza = ['finess', 'zone']
cols_pdmza.extend(['A{}'.format(i) for i in range(1, 6)])
cols.append(cols_pdmza)
exceptions.append([])

###########################################################################


hospidiag = generate_hospidiag(cols, exceptions)
le = pre_treatment()

for source in sources:
    print('Starting to process :', source)
    # Avoid Finess being read with several dtypes.
    data = pd.read_csv(source, sep=';', dtype={'Finess': str})

    is_test = 'test' in source

    # Test data has one more col, id, remove it.
    if is_test:
        data.drop(['id'], axis=1, inplace=True)

    columns = ['eta', 'nom_eta', 'prov_patient', 'dom_acti', 'age',
               'nombre_sej_ald', 'nombre_sej', 'an']

    if not is_test:
        columns.append('label')

    data.columns = columns

    # Data munging and cleaning for chronic disease reports
    # We only change eta after merging as it is one of the columns to join on.
    data.dom_acti = data.dom_acti.apply(lambda s: int(s[1:3]))
    data.prov_patient = data.prov_patient.apply(lambda s: s.replace('Inconnu', '0').replace('2A', '2000').replace('2B', '2001'))
    data.prov_patient = data.prov_patient.apply(lambda s: int(s.split('-')[0]))
    data.age = data.age.apply(lambda s: int(s == '>75 ans'))

    # Merging.
    print("Shape before merging :", data.shape)
    data = pd.merge(data, hospidiag, how='left', on=['eta', 'an'])
    print("Shape after merging :", data.shape)

    # Changing eta
    data.eta = data.eta.apply(lambda s: int(str(s).replace('2A', '2000').replace('2B', '2001')))
    data['nom_eta'] = le.transform(data.nom_eta)

    # Export to csv
    suffix = 'test' if is_test else 'data'
    data.to_csv('proc_{}.csv'.format(suffix), index=False)
