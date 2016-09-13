"""
Transform data so it is easier on ram.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data2.csv', sep=';')

data.columns = ['eta', 'nom_eta', 'prov_patient', 'dom_acti', 'age',
                'nombre_sej_ald', 'nombre_sej', 'an', 'label']


# Just manipulate for treatment
data.dom_acti = data.dom_acti.apply(lambda s: int(s[1:3]),
                                            convert_dtype=True)
data.prov_patient = data.prov_patient.apply(lambda s: s.replace('Inconnu', '0').replace('2A', '2000').replace('2B', '2001'))
data.prov_patient = data.prov_patient.apply(lambda s: int(s.split('-')[0]),
                                    convert_dtype=True)
data.age = data.age.apply(lambda s: int(s == '>75 ans'))
le = LabelEncoder()
data.nom_eta = le.fit_transform(data.nom_eta)
data.to_csv('proc_data.csv', index=False)
