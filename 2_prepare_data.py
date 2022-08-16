""""
The aim of the script is to prepare data for training

The script:

1. cleans and recodes data
2. creates new features
3. imputes missing data
4. remove uninformative features
5. makes dummmy variables
6. scales variables
7. Output:
    1. players_data_clean.csv
    2. match_data_clean.csv (there are made points 1-2 only)
    3. train_data_clean.csv
"""
#_________________________________________________import libraries______________________________________________________

import io, os, sys, types, time, datetime, math, random, requests, subprocess, tempfile

# Data Manipulation
import numpy as np
import pandas as pd
import re

# Visualization
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Feature Selection and Encoding
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

#_________________________________________________load data_____________________________________________________________

raw = pd.read_csv('raw_data.csv')
team = 'Devka Warszawa'

#_________________________________________________data cleaning_________________________________________________________

#remove unused column, remove duplicates, clean mvp column, clean scores column
raw = raw.drop('Unnamed: 0', axis='columns')
raw =  raw.drop_duplicates(['time_date', 'name'])

raw['mvp'] = raw['mvp'].apply(lambda x : 1 if x == '<span class="new_ico_other_star"></span>' else x == 0)
raw['mvp']= raw['mvp'].astype('int32')

raw['goals'] = raw['scores'].apply(lambda x : re.search(r'\d+', x))
raw['goals'] = raw['goals'].apply(lambda x : x == 0 if x == None else x.group())
raw['goals'] = raw['goals'].astype('int32')
raw = raw.drop('scores', axis='columns')

raw['man_on_pitch'] = raw['man_on_pitch'].apply(lambda x : x.split(' ',1)[0])
raw['man_on_pitch'] = raw['man_on_pitch'].astype('int32')

raw['match_lenght'] = raw['match_lenght'].replace(regex=['2 x ', ' min'], value='')
raw['match_lenght'] = raw['match_lenght'].astype('int32')

raw['result_input'] = raw['result_input'].replace(regex=["b'", "'"], value='')

#place analyzed team and opponent team data in distinct columns

raw['opponent_team_name'] = raw.apply(lambda x : x['team_r_name'] if x['team_l_name'] == 'Devka Warszawa' else x['team_l_name'], axis = 1)

raw['analyzed team_result'] = raw.apply(lambda x : x['team_l_score'] if x['team_l_name'] == 'Devka Warszawa' else x['team_r_score'], axis = 1)
raw['opponent_team_result'] = raw.apply(lambda x : x['team_l_score'] if x['team_l_name'] != 'Devka Warszawa' else x['team_r_score'], axis = 1)

raw['analyzed team_rank'] = raw.apply(lambda x : x['team_l_rank'] if x['team_l_name'] == 'Devka Warszawa' else x['team_r_rank'], axis = 1)
raw['opponent_team_rank'] = raw.apply(lambda x : x['team_l_rank'] if x['team_l_name'] != 'Devka Warszawa' else x['team_r_rank'], axis = 1)

raw = raw.drop(['team_l_name', 'team_l_rank', 'team_l_score', 'team_r_name', 'team_r_rank', 'team_r_score'], axis='columns')

# create new features
raw['weekday'] = raw.apply(lambda x : x['weekday'] if x['weekday']  in ['Ndz', 'Sob'] else 'Pon_Pt', axis = 1)

raw['time'] = raw['time'].replace(regex=[':'], value='')
raw['time'] = raw['time'].astype('int32')
raw['time'] = raw.apply(lambda x : x['time'] if x['time'] >= 800  else x['time'] + 1200, axis = 1)
raw['time'] = pd.cut(np.array(raw['time']), bins=[0, 1100, 1400, 1800, 2000], labels=['morning', 'noon', 'afternoon', 'evening'])

raw['pos_d'] = raw.apply(lambda x : 'a_' + x['position'] if x['teams'] == team else 'o_' + x['position'], axis = 1)
raw['mvp_d'] = raw.apply(lambda x : x['name'] if x['mvp'] == 1 else 'no_mvp', axis = 1)
raw['goals_d'] = raw.apply(lambda x : x['name'] if x['goals'] > 0 else 'no_goals', axis = 1)
raw['name'] = raw.apply(lambda x : 'a_' + x['name'] if x['teams'] == team else 'o_' + x['name'], axis = 1)

#_________________________________________________matches data frame____________________________________________________

raw1 = pd.get_dummies(raw, columns = ['pos_d','name', 'mvp_d', 'goals_d'])

raw1 = raw1.drop(['position', 'teams', 'goals'], axis='columns')

raw2 = raw1.groupby(['time_date'])[raw1.loc[:, 'pos_d_a_Bramkarz':].columns].sum()

raw3 = pd.DataFrame(raw1.groupby(['time_date'])['mvp'].sum())

raw4 = raw1[['time_date', 'date', 'league', 'man_on_pitch', 'match_lenght', 'stadium', 'time', 'weekday',
             'opponent_team_name', 'analyzed team_result',
            'opponent_team_result', 'analyzed team_rank', 'opponent_team_rank']]
raw4 = raw4.drop_duplicates('time_date')

raw5 = raw4.merge(raw2, how= 'left', on= 'time_date', validate='one_to_one')
raw6 = raw5.merge(raw3, how= 'left', on= 'time_date', validate='one_to_one')

raw6['withdrawal'] = raw6.apply(lambda x :  1 if x['mvp'] == 0 and abs(x['analyzed team_result'] - x['opponent_team_result']) == 10  else  0, axis = 1)
raw6['mmpd'] =raw6.duplicated('date')
raw6['win'] = raw6.apply(lambda x :  1 if x['analyzed team_result'] - x['opponent_team_result'] > 0  else  0, axis = 1)
raw6['pos_d_a_Zawodnicy bez pozycji'] = raw6.apply(lambda x :  x['pos_d_a_Zawodnicy bez pozycji'] if 'pos_d_a_Zawodnicy bez pozycji' in raw6.columns else  0, axis = 1)
raw6['n_ply_a'] = raw6[['pos_d_a_Bramkarz', 'pos_d_a_Napastnik', 'pos_d_a_Obrońca', 'pos_d_a_Pomocnik', 'pos_d_a_Zawodnicy bez pozycji']].sum(axis=1)
raw6['n_ply_o'] = raw6[['pos_d_o_Bramkarz', 'pos_d_o_Napastnik', 'pos_d_o_Obrońca', 'pos_d_o_Pomocnik', 'pos_d_o_Zawodnicy bez pozycji']].sum(axis=1)

raw6['date'] = pd.to_datetime(raw6['date'], dayfirst=True)
raw6['match'] = raw6.apply(lambda x : x['date'].strftime("%Y-%m-%d") + "_" + x['opponent_team_name'], axis = 1)


raw7 = pd.get_dummies(raw6, columns = ['stadium','league', 'weekday','time', 'opponent_team_name',
                                       'man_on_pitch', 'match_lenght'])


raw8 = raw7.drop(['time_date', 'analyzed team_result','opponent_team_result', 'mvp'], axis='columns')

raw9 = raw6.set_index('match')

raw9.to_csv('match_data_clean')


#_________________________________________________preparing data for modeling___________________________________________

#remove features which:
# are not known before match - they do not help to predict the result: goals, mvp
# have less than 10 occurences of every class unless they are with potential to be informative. These features are stored
# in col_not_to_drop variable

col_not_to_drop = ['match', 'analyzed team_rank','opponent_team_rank', 'pos_d_a_Bramkarz', 'pos_d_a_Napastnik',
                   'pos_d_a_Obrońca', 'pos_d_a_Pomocnik', 'pos_d_o_Bramkarz',
'pos_d_o_Napastnik', 'pos_d_o_Obrońca', 'pos_d_o_Pomocnik', 'pos_d_o_Zawodnicy bez pozycji', 'pos_d_o_Zawodnicy bez pozycji',
'withdrawal', 'win']

for cols in raw8.columns:
    if cols not in col_not_to_drop:
        if any(raw8[cols].value_counts() < 10) or cols[:5] == 'goals' or cols[:3] == 'mvp':
            raw8.drop(cols, axis=1, inplace= True)

#remove withdrawals - we do not want them to interfere

raw10 = raw8[raw8["withdrawal"] == 0]

raw10 = raw8.drop("withdrawal", axis= 1)

#set index
raw10 = raw10.set_index('match')

#all features should be numeric check and then make this happen
raw10 = raw10.apply(pd.to_numeric, errors='coerce')


#standarization do not use for dummies

to_standard_cols = ['pos_d_a_Bramkarz', 'pos_d_a_Napastnik',
                   'pos_d_a_Obrońca', 'pos_d_a_Pomocnik', 'pos_d_o_Bramkarz',
'pos_d_o_Napastnik', 'pos_d_o_Obrońca', 'pos_d_o_Pomocnik', 'pos_d_o_Zawodnicy bez pozycji', 'pos_d_o_Zawodnicy bez pozycji']

# apply standardization on numerical features
for i in to_standard_cols:
    scale = StandardScaler().fit(raw10[[i]])
    raw10[i] = scale.transform(raw10[[i]])

#missing values imputation
to_impute_cols = ['analyzed team_rank','opponent_team_rank']

#for pandas’ dataframes with nullable integer dtypes with missing values, missing_values should be set to np.nan, since pd.NA will be converted to np.nan.

raw10[to_impute_cols] = raw10[to_impute_cols].apply(pd.to_numeric, errors='coerce')

imputer = KNNImputer()
for i in to_impute_cols:
    raw10[i] = imputer.fit_transform(raw10[[i]])

#standarization of columns with imputed columns

for i in to_impute_cols:
    scale = StandardScaler().fit(raw10[[i]])
    raw10[i] = scale.transform(raw10[[i]])

raw10.to_csv('train_data_clean')

#_________________________________________________EDA___________________________________________________________________
raw10.describe()

# Let’s plot the distribution of each feature
def plot_distribution(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if dataset.dtypes[column] == np.object and dataset.columns:
            g = sns.countplot(y=column, data=dataset)
            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            plt.xticks(rotation=25)
        else:
            g = sns.distplot(dataset[column])
            plt.xticks(rotation=25)


plot_distribution(raw10, cols=8, width=20, height=20, hspace=0.45, wspace=0.5)


# Plot a count of the categories from each categorical feature split by our prediction class: salary - predclass.
def plot_bivariate_bar(dataset, hue, cols=8, width=20, height=15, hspace=0.2, wspace=0.5):
    dataset = dataset.select_dtypes(include=[np.object])
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if dataset.dtypes[column] == np.object:
            g = sns.countplot(y=column, hue=hue, data=dataset)
            substrings = [s.get_text()[:10] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)


plot_bivariate_bar(raw10, hue='win', cols=8, width=20, height=12, hspace=0.4, wspace=0.5)

sns.set_theme(style="white")
# Compute the correlation matrix
corr = raw10.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


#_________________________________________________players data frame____________________________________________________

df = raw[raw['teams'] == team]

#remove withdrawals
#list withdrawals
withdrawal_indexes = list(raw8[raw8['withdrawal'] == 1].index)

df['withdrawal'] = df['time_date'].apply(lambda x : 1 if x in withdrawal_indexes else x == 0)
df = df[df['withdrawal'] == 0]

df['win'] = df.apply(lambda x :  1 if x['analyzed team_result'] - x['opponent_team_result'] > 0  else  0, axis = 1)
df['draw'] = df.apply(lambda x :  1 if x['analyzed team_result'] - x['opponent_team_result'] == 0  else  0, axis = 1)
df['lose'] = df.apply(lambda x :  1 if x['analyzed team_result'] - x['opponent_team_result'] < 0  else  0, axis = 1)

df['result'] = df['analyzed team_result'] - df['opponent_team_result']

df['date'] = pd.to_datetime(df['time_date'].apply(lambda x : x[2:12]), dayfirst=True)

df['season'] = df['date'].apply(
    lambda x: '/'.join([str(x.year), str(x.year + 1)[-2:]]) if int(x.strftime('%j')) >= 243 else '/'.join(
        [str(x.year - 1), str(x.year)[-2:]]))

conditions = [
    (df['league'] == 'no league') & (df['mvp'] == 0),
    (df['league'] == 'no league') & (df['mvp'] == 1),
    (df['league'] != 'no league') & (df['mvp'] == 0) & (df['result'] < 0),
    (df['league'] != 'no league') & (df['mvp'] == 1) & (df['result'] < 0),
    (df['league'] != 'no league') & (df['mvp'] == 0) & (df['result'] == 0),
    (df['league'] != 'no league') & (df['mvp'] == 1) & (df['result'] == 0),
    (df['league'] != 'no league') & (df['mvp'] == 0) & (df['result'] > 0),
    (df['league'] != 'no league') & (df['mvp'] == 1) & (df['result'] > 0)
    ]
choices = [1, 3, 1, 3, 2, 4, 4, 6]
df['points'] = np.select(conditions, choices, default=1)

df = df[['name', 'position', 'mvp', 'goals', 'date', 'win', 'lose', 'draw', 'result', 'opponent_team_result', 'points', 'season']]

dfp = df.groupby('name').agg(
    position = pd.NamedAgg(column='position', aggfunc=lambda x: x.value_counts().index[0]),
    mvp = pd.NamedAgg(column='mvp', aggfunc='sum'),
    goals = pd.NamedAgg(column='goals', aggfunc='sum'),
    app = pd.NamedAgg(column='name', aggfunc='size'),
    points = pd.NamedAgg(column='points', aggfunc='sum'),
    last_match = pd.NamedAgg(column='date', aggfunc= 'max'),
    first_match = pd.NamedAgg(column='date', aggfunc= 'min'),
    days = pd.NamedAgg(column='date', aggfunc= lambda x: (max(x)-min(x)).days),
    wins = pd.NamedAgg(column='win', aggfunc= 'sum'),
    draws = pd.NamedAgg(column='draw', aggfunc= 'sum'),
    losses = pd.NamedAgg(column='lose', aggfunc= 'sum'),
    balance = pd.NamedAgg(column='result', aggfunc= 'sum'),
    clean_sheets = pd.NamedAgg(column='opponent_team_result', aggfunc= lambda x: sum(1 for y in x if y > 2))
)

dfp = dfp[dfp['last_match'] != dfp['first_match']]

dfp.to_csv('players_data_clean')