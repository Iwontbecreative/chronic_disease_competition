import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="scale predictions")
parser.add_argument('filename', type=str, help='Filename to scale')
args = parser.parse_args()

FILENAME = args.filename

average_year = {
        2008: 0.1027,
        2009: 0.1129,
        2010: 0.1258,
        2011: 0.1317,
        2012: 0.1384,
        2013: 0.1483,
        2014: 0.161,
        2015: 0.16,
        }

magic_number = 328679

data = pd.read_csv(FILENAME, sep=';')
print('Old average was:', data.cible.mean())

y2014 = data.loc[:magic_number]
y2015 = data.loc[magic_number + 1:]
y2014.cible *= average_year[2014] / y2014.cible.mean()
y2015.cible *= average_year[2015] / y2015.cible.mean()

new_data = pd.concat([y2014, y2015])
print('New average is:', new_data.cible.mean())

new_data.to_csv(FILENAME[:-4] + '_scaled.csv', sep=';', index=False)

