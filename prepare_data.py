#import libraries
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

#import data
raw = pd.read_csv('raw_data.csv')

team_a = 'Devka Warszawa'

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

raw['opponent_team_name'] = raw.apply(lambda x : x['team_r_name'] if x['team_l_name'] == 'Devka Warszawa' else x['team_l_name'], axis = 1)

raw['analyzed team_result'] = raw.apply(lambda x : x['team_l_score'] if x['team_l_name'] == 'Devka Warszawa' else x['team_r_score'], axis = 1)
raw['opponent_team_result'] = raw.apply(lambda x : x['team_l_score'] if x['team_l_name'] != 'Devka Warszawa' else x['team_r_score'], axis = 1)

raw['analyzed team_rank'] = raw.apply(lambda x : x['team_l_rank'] if x['team_l_name'] == 'Devka Warszawa' else x['team_r_rank'], axis = 1)
raw['opponent_team_rank'] = raw.apply(lambda x : x['team_l_rank'] if x['team_l_name'] != 'Devka Warszawa' else x['team_r_rank'], axis = 1)

raw['weekday'] = raw.apply(lambda x : x['weekday'] if x['weekday']  in ['Ndz', 'Sob'] else 'Pon_Pt', axis = 1)

raw['time'] = raw['time'].replace(regex=[':'], value='')
raw['time'] = raw['time'].astype('int32')
raw['time'] = raw.apply(lambda x : x['time'] if x['time'] >= 800  else x['time'] + 1200, axis = 1)
raw['time'] = pd.cut(np.array(raw['time']), bins=[0, 1100, 1400, 1800, 2000], labels=['morning', 'noon', 'afternoon', 'evening'])

raw = raw.drop(['team_l_name', 'team_l_rank', 'team_l_score', 'team_r_name', 'team_r_rank', 'team_r_score'], axis='columns')

raw['pos_d'] = raw.apply(lambda x : 'a_' + x['position'] if x['teams'] == team_a else 'o_' + x['position'], axis = 1)
raw['mvp_d'] = raw.apply(lambda x : x['name'] if x['mvp'] == 1 else 'no_mvp', axis = 1)
raw['goals_d'] = raw.apply(lambda x : x['name'] if x['goals'] > 0 else 'no_goals', axis = 1)

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

raw7 = pd.get_dummies(raw6, columns = ['stadium','league', 'weekday','time', 'opponent_team_name',
                                       'man_on_pitch', 'match_lenght'])
raw8 = raw7.drop(['date', 'analyzed team_result','opponent_team_result', 'mvp'], axis='columns')


raw9 = raw6.set_index('time_date')
raw9.to_csv('match_data_clean')





col_not_to_drop = ['time_date', 'analyzed team_rank','opponent_team_rank', 'pos_d_a_Bramkarz', 'pos_d_a_Napastnik',
                   'pos_d_a_Obrońca', 'pos_d_a_Pomocnik', 'pos_d_o_Bramkarz',
'pos_d_o_Napastnik', 'pos_d_o_Obrońca', 'pos_d_o_Pomocnik', 'pos_d_o_Zawodnicy bez pozycji', 'pos_d_o_Zawodnicy bez pozycji',
'withdrawal', 'win']

for cols in raw8.columns:
    if cols not in col_not_to_drop:
        if any(raw8[cols].value_counts() < 10):
            raw8.drop(cols, axis=1, inplace= True)

raw8 = raw8[raw8["withdrawal"] == 0]
raw8 = raw8.drop("withdrawal", axis= columns)
raw8 = raw8.set_index('time_date')

#standarization do not use for dummies

to_standard_cols = ['pos_d_a_Bramkarz', 'pos_d_a_Napastnik',
                   'pos_d_a_Obrońca', 'pos_d_a_Pomocnik', 'pos_d_o_Bramkarz',
'pos_d_o_Napastnik', 'pos_d_o_Obrońca', 'pos_d_o_Pomocnik', 'pos_d_o_Zawodnicy bez pozycji', 'pos_d_o_Zawodnicy bez pozycji']

# apply standardization on numerical features
for i in to_standard_cols:
    scale = StandardScaler().fit(raw8[[i]])
    raw8[i] = scale.transform(raw8[[i]])

#missing values imputation
to_impute_cols = ['analyzed team_rank','opponent_team_rank']

#for pandas’ dataframes with nullable integer dtypes with missing values, missing_values should be set to np.nan, since pd.NA will be converted to np.nan.

raw8[to_impute_cols] = raw8[to_impute_cols].apply(pd.to_numeric, errors='coerce')

imputer = KNNImputer()
for i in to_impute_cols:
    raw8[i] = imputer.fit_transform(raw8[[i]])

#standarization of columns with imputed columns

for i in to_impute_cols:
    scale = StandardScaler().fit(raw8[[i]])
    raw8[i] = scale.transform(raw8[[i]])

raw8.to_csv('train_data_clean')
