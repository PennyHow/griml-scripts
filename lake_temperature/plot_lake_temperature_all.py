#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:26:49 2024

@author: pho
"""

import glob,re, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import matplotlib.lines as mlines

# Define files
iiml_file = 'ALL-ESA-GRIML-IML-MERGED-fv2.shp'
indir = 'ST_lakes_2016-2023_inventory_100m_buffer/'

infile = glob.glob(indir+'*.csv')

# Check for missing lakes
num = [int(re.findall(r'\d+',str(Path(f).stem))[0]) for f in list(infile)]
res = [ele for ele in range(max(num)+1) if ele not in num]
print('Missing lake ids : ' + str(res))

# Load ice marginal lake inventory
iiml = gpd.read_file(iiml_file)
# iiml = iiml.dissolve(by='lake_id')
iiml["centroid"] = iiml["geometry"].centroid

# Define empty df for concat
dfs=[]
dfs_st=[]
dfs_min=[]
dfs_max=[]

# Prime plot
fig, ax = plt.subplots(1, 1, figsize=(10,8), sharex=True)
fsize1 = 16
fsize2 = 14
fsize3 = 12
fsty = 'arial'
pad=1.
lloc=4
lw=1.5

# Iterate over lake ST files
for f in list(infile):
    df = pd.read_csv(f, comment='#', index_col=2,
                      na_values=['','nan'], parse_dates=True,
                      sep=',', 
                      skip_blank_lines=True) 


    df = df.drop(df[df['ST'] < 0].index)

    # Filter to inventory years
    df['year'] = df.index.year
    # df = df.loc[(df['year'] >= 2016) & (df['year'] <= 2023)]

    # Filter to month of August
    df['month'] = df.index.month
    df = df.loc[(df['month'] >= 8) & (df['month'] <= 8)]
    
    # Reformat columns
    st_count=[]
    st_max=[]
    st_min=[]
    st_stddev=[]
    
    for index, row in df.iterrows():
        
        st_count.append(int(re.findall(r'\b\d+\b', row['ST_count'])[0]))
        
        if not row['ST_max'] in ['[null]','NaN']:
            st_max.append(float(row['ST_max'][1:-1]))
        else:
            st_max.append(np.nan)
            
        if not row['ST_min'] in ['[null]','NaN']:
            st_min.append(float(row['ST_min'][1:-1]))
        else:
            st_min.append(np.nan)
         
        if not row['ST_stddev'] in ['[null]','NaN']:
            st_stddev.append(float(row['ST_stddev'][1:-1]))
        else:
            st_stddev.append(np.nan)
            
    df['ST_count_int'] = st_count
    df['ST_max_int'] = st_max
    df['ST_min_int'] = st_min
    df['ST_stddev_int'] = st_stddev
                        
    if len(df)>1:
        
        for i in list(set(df.lake_id)):
            df_set = df.loc[(df['lake_id'] == i)]
            
            # Sort by index and re-define satellite description
            df_set = df_set.sort_index()
            df_set['LC'] = [d[d.index('L'):d.index('L')+4] for d in list(df_set['system:index'])]
            
            # Define drainage basin
            lake_id = int(list(df_set['lake_id'])[0])
            i = iiml.loc[iiml.index == lake_id]  
            
            # Resample to yearly means
            df_st = df_set['ST'].resample('YS').mean()
            df_st = df_st.set_axis(df_st.index.year)
            df_stmin = df_set['ST_min_int'].resample('YS').min()
            df_stmin = df_stmin.set_axis(df_stmin.index.year)
            df_stmax = df_set['ST_max_int'].resample('YS').max()
            df_stmax = df_stmax.set_axis(df_stmax.index.year)
            
            # Plot based on basin
            ax.plot(list(df_st.index), list(df_st), c='#B6B6B6', linewidth=0.1)
    
            dfs.append(df_set)
            dfs_st.append(df_st)
            dfs_min.append(df_stmin)
            dfs_max.append(df_stmax)


dfs_st = pd.concat(dfs_st)        
dfs_min = pd.concat(dfs_min)
dfs_max = pd.concat(dfs_max)

dfs_st.index = pd.to_datetime(dfs_st.index, format='%Y')
dfs_min.index = pd.to_datetime(dfs_min.index, format='%Y')
dfs_max.index = pd.to_datetime(dfs_max.index, format='%Y')

st_mean = dfs_st.resample('YS').mean()
st_min = dfs_min.resample('YS').min()
st_max = dfs_max.resample('YS').max()

ax.plot(list(st_mean.index.year), list(st_mean), c='k', linewidth=3)
# ax.plot(list(st_mean.index.year), list(st_max), c='k', linestyle='--', linewidth=3)

labels = [str(round(s,1)) for s in st_mean]
# for a in range(len(labels))[::2]:
for a in range(len(labels)):
    ax.annotate(labels[a], xy=(list(st_mean.index.year)[a], list(st_mean)[a]+0.6))
  
ax.set_ylim([0, 15])
ax.set_xlim([2016,2023])

ax.set_ylabel('Estimated lake surface temperature $^\circ$C', fontsize=fsize1)#, labelpad=5)
ax.set_xlabel('Inventory year', fontsize=fsize1, labelpad=5)

lines = [mlines.Line2D([], [], color='k', label='August average of all lakes'),
         mlines.Line2D([],[],color='#B6B6B6', label='August average of individual lake')]
ax.legend(loc=4, handles=lines, fontsize=fsize2)

# text=''
# for a,b in zip(list(st_mean.index.year), list(st_mean)):
#     text=text+str(a)+': '+str(round(b, 2))+'\n'
# props = dict(boxstyle='round', facecolor='#3F8BD7', alpha=0.2)
# ax.text(1.05, 0, text, fontsize=fsize2, horizontalalignment='center', 
#         bbox=props, transform=ax.transAxes)       
 
plt.savefig('lake_temp_all_2016_2023.png',dpi=300)
# plt.show()
    
