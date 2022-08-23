#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 19:48:49 2022

@author: alberto
"""

import pandas as pd
import numpy as np

df = pd.read_csv('countries.csv')

'''
PREGUNTA 1
df2 = df[df.country == 'Tunisia'].continent
df2_list = df2.values.tolist()[0]
df2_list

PREGUNTA 2
df2 = df[(df.lifeExp > 80)&(df.year == 2007)]
df2.loc[:,'country']

PREGUNTA 3
df2 = df[(df.continent == 'Americas')]
df2 = df2.sort_values('gdpPercap')
df2['country'].iloc[-1]

PREGUNTA 4
df_venezuela = df[(df.country == 'Venezuela') & (df.year == 1967)]
df_paraguay = df[(df.country == 'Paraguay') & (df.year == 1967)]
df2 = pd.concat([df_venezuela, df_paraguay], ignore_index = True, axis = 0)
df2 = df2.sort_values('pop')
df2['country'].iloc[-1]

PREGUNTA 5
df2 = df[(df.country == 'Panama') & (df.lifeExp > 60)]
df2['year'].iloc[0]
Out[7]: 1962

PREGUNTA 6
df2 = df[(df.continent == 'Africa') & (df.year == 2007)]
df2['lifeExp'].mean()
Out[102]: 54.80603846153845

PREGUNTA 7
df_2002 = df[(df['year'] == 2002)]
df_2007 = df[(df['year'] == 2007)]
df_2002 = df_2002.reset_index(drop=True)
df_2007 = df_2007.reset_index(drop=True)
df_2002['less_gdp'] = np.where(df_2002['gdpPercap'] > df_2007['gdpPercap'], 'True', 'False')
df2 = df_2002[(df_2002['less_gdp'] == 'True')]
df2['country']


PREGUNTA 8
df2 =df[(df.year == 2007)]
df2 =df2.sort_values('pop')
df2['country'].iloc[-1]
Out[115]: 'China'

PREGUNTA 9
df2 = df[(df.continent == 'Americas') & (df.year == 2007)]
df2['pop'].sum()
Out[120]: 898871184.0

PREGUNTA 10
df_africa   = df[(df['continent'] == 'Africa'  ) & (df['year'] == 2007)]
df_asia     = df[(df['continent'] == 'Asia'    ) & (df['year'] == 2007)]
df_americas = df[(df['continent'] == 'Americas') & (df['year'] == 2007)]
df_europe   = df[(df['continent'] == 'Europe'  ) & (df['year'] == 2007)]
df_oceania  = df[(df['continent'] == 'Oceania' ) & (df['year'] == 2007)]

data_africa = {'continent' : df_africa['continent'].iloc[0],                                      
   'pop':df_africa['pop'].sum() }

data_asia = {'continent':df_asia['continent'].iloc[0], 
 'pop':df_asia['pop'].sum()}

data_americas = {'continent':df_americas['continent'].iloc[0],     
 			'pop':df_americas['pop'].sum()}

data_europe = {'continent':df_europe['continent'].iloc[0], 
   'pop':df_europe['pop'].sum()}

data_oceania = {'continent':df_oceania['continent'].iloc[0],
    'pop':df_oceania['pop'].sum()}

df2 = pd.DataFrame()
df2['continent'] = None
df2['pop'] = None

df2 = df2.append(data_africa,   ignore_index = True)
df2 = df2.append(data_asia, 	  ignore_index = True)
df2 = df2.append(data_americas, ignore_index = True)
df2 = df2.append(data_europe,   ignore_index = True)
df2 = df2.append(data_oceania,  ignore_index = True)

df2 = df2.sort_values('pop')
df2['continent'].iloc[0]
Out[50]: 'Oceania'

PREGUNTA 11
df2 = df[(df.continent == 'Europe')]
df2['gdpPercap'].mean()
Out[107]: 14469.475533302237

PREGUNTA 12
df2 = df[(df.continent == 'Europe') & (df['pop'] > 70000000.0)]
df2['country'].iloc[0]
Out[15]: 'Germany'

'''

