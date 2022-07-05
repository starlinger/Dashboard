import pandas as pd
import numpy as np
from scipy import stats


def get_pos_no_rep(path='/assets/data/2022-03-25_norep_all.csv', reduced =  False):

    df = pd.read_csv(path, sep=';')
    df = df.iloc[:,np.r_[1:2, 4:5, 17:26]]  #für virus positive aus no_rep_all
    #print(df[:5])
    #group 1 and 2 = 0, group 4 = 1
    #print(df.shape[0])
    df.loc[df['Group'] == 1, 'Group'] = 0
    df.loc[df['Group'] == 2, 'Group'] = 0
    df.loc[df['Group'] == 4, 'Group'] = 1

    #drop group 3 and 5
    #df = df[df['Group'] != 3]
    df = df[df['Group'] != 5]

    #nur viruspositive (Virus: 4 = B19V)
    df = df[df['Virus: 1=Cox, 4=B19V, 6=HHV6, 5=EBV, 2=ADV (including  double infections) '] == 4]
    df = df.drop('Virus: 1=Cox, 4=B19V, 6=HHV6, 5=EBV, 2=ADV (including  double infections) ', axis = 1)

    #deletes nans
    df = df.dropna(how='any')
    if reduced: df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    print('Samples: ' + str(df.shape[0]) + ' Group0 ' + str(df['Group'].value_counts()[0])  + ' Group1 ' + str(df['Group'].value_counts()[1]))
    print(df.columns)
    return df


def get_virus_negative(path='/assets/data/2022-03-25_virusneg.csv', reduced =  False):
    df = pd.read_csv(path, sep=';')
    df = df.iloc[:,np.r_[1:2, 4:5, 17:25]]   #for virusnegative and rep_all
    
    #group 1 and 2 = 0, group 4 = 1
    #print(df.shape[0])
    df.loc[df['Group'] == 1, 'Group'] = 0
    df.loc[df['Group'] == 2, 'Group'] = 0
    df.loc[df['Group'] == 4, 'Group'] = 1

    #drop group 3 and 5
    #df = df[df['Group'] != 3]
    df.loc[df['Group'] == 3, 'Group'] = 1
    df = df[df['Group'] != 5]
    
    #deletes nans
    df = df.dropna(how='any')
    if reduced: df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    print('Samples: ' + str(df.shape[0]) + ' Group0 ' + str(df['Group'].value_counts()[0])  + ' Group1 ' + str(df['Group'].value_counts()[1]))
    print(df.columns)
    return df

def get_pos_rep(path='/assets/data/2022-05-30-B19_pos.csv', reduced =  False):
    df = pd.read_csv(path, sep=';')
    df.rename(columns={'EMB: CD45R0 (numeric) ':'EMB: CD45 (numeric) '}, inplace=True)
    df = df.iloc[:,np.r_[1:2, 4:5, 16:25]]    #for viruspositive

    #group 1 and 2 = 0, group 4 = 1
    #print(df.shape[0])
    df.loc[df['Group'] == 1, 'Group'] = 0
    df.loc[df['Group'] == 2, 'Group'] = 0
    df.loc[df['Group'] == 4, 'Group'] = 1

    #drop group 3 and 5
    #df = df[df['Group'] != 3]
    df = df[df['Group'] != 5]

    #nur replikation (Virus: 4 = B19V)
    df = df[df['B19V mRNA   yes1/no0'] == 1]
    df = df.drop('B19V mRNA   yes1/no0', axis = 1)

    df = df.dropna(how='any')
    if reduced: df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    print('Samples: ' + str(df.shape[0]) + ' Group0 ' + str(df['Group'].value_counts()[0])  + ' Group1 ' + str(df['Group'].value_counts()[1]))
    print(df.columns)
    return df

def get_no_rep_all(path='/assets/data/2022-03-25_norep_all.csv', reduced =  False):
    df = pd.read_csv(path, sep=';')

    df = df.iloc[:,np.r_[1:2, 4:5, 17:25]]   #for virusnegative and rep_all

    #group 1 and 2 = 0, group 4 = 1
    print(df.shape[0])
    df.loc[df['Group'] == 1, 'Group'] = 0
    df.loc[df['Group'] == 2, 'Group'] = 0
    df.loc[df['Group'] == 4, 'Group'] = 1

    #drop group 3 and 5
    #df = df[df['Group'] != 3]
    df = df[df['Group'] != 5]

    df = df.dropna(how='any')
    if reduced: df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    print('Samples: ' + str(df.shape[0]) + ' Group0 ' + str(df['Group'].value_counts()[0])  + ' Group1 ' + str(df['Group'].value_counts()[1]))
    print(df.columns)
    return df