"""
Columns to keep in the algorithm
"""

# Columns to keep. Note that keeping 40+ columns will usually swap (on 4Gb ram)
# using Random Forest Regressors or Gradient Boosted Trees.
to_keep = ['eta', 'nom_eta', 'prov_patient', 'dom_acti', 'age',
           'nombre_sej_ald', 'nombre_sej', 'an', 'id', 'label']
to_keep.extend(['A9', 'A12', 'A13', 'A14', 'CI_AC1',
                'CI_AC6', 'CI_RH4', 'CI_RH1', 'CI_RH3',
                'CI_A5', 'CI_A12', 'CI_A15', 'P2', 'P13', 'P9',
                'P12', 'P14', 'A1bis', 'A4bis',
                'A5bis', 'cat'])
to_keep.extend(['RH{}'.format(i) for i in range(2, 6)])

to_keep_2015 = ['eta', 'nom_eta', 'prov_patient', 'dom_acti', 'age',
                'nombre_sej_ald', 'nombre_sej', 'an', 'id', 'label']
to_keep_2015.extend(['A1bis', 'A2bis', 'A4bis', 'A5bis', 'cat',
                     'A9', 'A12', 'CI_A5', 'CI_A12', 'CI_A15', 'P2',
                     'P13', 'P12', 'CI_A16_6'])
