import pandas as pd
from glob import glob


average_year = {
        2008: 0.2306,
        2009: 0.2513,
        2010: 0.2762,
        2011: 0.2879,
        2012: 0.2999,
        2013: 0.315,
        2014: 0.3382,
        2015: 0.3359,
        }


ans = pd.read_csv('proc_data.csv', usecols=['an'])
mult = ans.an.apply(lambda y: average_year[y])

for name in glob('first_level/OLD/*train*.csv'):
    data = pd.read_csv(name, sep=';')
    local_average_y = {}
    for y in ans.an.unique():
        local_average_y[y] = data[ans.an == y].cible.mean()
    mult_2 = mult / ans.an.apply(lambda y: local_average_y[y])
    data.cible = data.cible * mult_2
    data.to_csv(name[:-4] + '_ms.csv', index=False, sep=';')


ans = pd.read_csv('proc_test.csv', usecols=['an'])
mult = ans.an.apply(lambda y: average_year[y])

for name in glob('first_level/OLD/*test*.csv'):
    data = pd.read_csv(name, sep=';')
    local_average_y = {}
    for y in ans.an.unique():
        local_average_y[y] = data[ans.an == y].cible.mean()
    mult_2 = mult / ans.an.apply(lambda y: local_average_y[y])
    data.cible = data.cible * mult_2
    data.to_csv(name[:-4] + '_ms.csv', index=False, sep=';')


