#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:26:49 2024

@author: pho
"""

import glob,re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
from pathlib import Path

# Define files
iiml_file = 'ALL-ESA-GRIML-IML-MERGED-fv2.shp'
indir = 'ST_lakes_2016-2023_inventory_2/'

infile = glob.glob(indir+'*.csv')

# Check for missing lakes
num = [int(re.findall(r'\d+',str(Path(f).stem))[0]) for f in list(infile)]
res = [ele for ele in range(max(num)+1) if ele not in num]
print('Missing lake ids : ' + str(res))

# Load ice marginal lake inventory
iiml = gpd.read_file(iiml_file)
# iiml = iiml.dissolve(by='LakeID')
iiml["centroid"] = iiml["geometry"].centroid

# Define basins
basins = ['SE','SW','CW','CE','NW','NO','NE']

df_se1=[]
df_sw1=[]
df_cw1=[]
df_ce1=[]
df_nw1=[]
df_no1=[]
df_ne1=[]
is_dfs=[df_se1, df_sw1, df_cw1, df_ce1, df_nw1, df_no1, df_ne1]

# Iterate over lake ST files
for f in list(infile):
    df = pd.read_csv(f, comment='#', index_col=2,
                      na_values=['','nan'], parse_dates=True,
                      sep=',', skip_blank_lines=True) 

    df = df.drop(df[df['ST'] < 0].index)

    # Filter to inventory years
    df['year'] = df.index.year
    df = df.loc[(df['year'] >= 2016) & (df['year'] <= 2023)]
    
    # Filter to months between June and September
    df['month'] = df.index.month
    df = df.loc[(df['month'] >= 8) & (df['month'] <= 8)]
        
    # Reformat columns
    st_count=[]
    st_max=[]
    st_min=[]
    st_stddev=[]
    
    for index, row in df.iterrows():
        
        st_count.append(int(re.findall(r'\b\d+\b', row['ST_count'])[0]))
        
        if len(re.findall(r"\d+\.\d+", row['ST_max']))>0:
            st_max.append(float(re.findall(r"\d+\.\d+", row['ST_max'])[0]))
        else:
            st_max.append(np.nan)
            
        if len(re.findall(r"\d+\.\d+", row['ST_min']))>0:
            st_min.append(float(re.findall(r"\d+\.\d+", row['ST_min'])[0]))
        else:
            st_min.append(np.nan)
            
        if len(re.findall(r"\d+\.\d+", row['ST_stddev']))>0:
            st_stddev.append(float(re.findall(r"\d+\.\d+", row['ST_stddev'])[0]))
        else:
            st_stddev.append(np.nan)
            
    df['ST_count_int'] = st_count
    df['ST_max_int'] = st_max
    df['ST_min_int'] = st_min
    df['ST_stddev_int'] = st_stddev
                        
    if len(df)>1:
        # Sort by index and re-define satellite description
        df = df.sort_index()
        df['LC'] = [d[d.index('L'):d.index('L')+4] for d in list(df['system:index'])]
        
        for i in list(set(df.lake_id)):
            df_set = df.loc[(df['lake_id'] == i)]        
        
            # Resample to yearly means
            df_y = df_set['ST'].resample(datetime.timedelta(days=365)).mean()
            
            # Define drainage basin
            lake_id = int(list(df_set['lake_id'])[0])
            i = iiml.loc[iiml.lake_id == lake_id]
            df_set['centroid'] = list(i['centroid'])[0]
            df_set['d_basin'] = list(i['region'])[0]
            df_set['d_margin'] = list(i['margin'])[0]
            
            # Plot based on basin
            b = [r for r in range(len(basins)) if list(df_set['d_basin'])[0] in basins[r]]

            is_dfs[b[0]].append(df_y)


for b in range(len(is_dfs)):
    aver_st = pd.concat(is_dfs[b])
    mean = aver_st.resample('YS').mean()

    print(basins[b])
    print('Mean lake temperature: ')
    # print(mean)
    print(np.mean(list(mean)))
    
