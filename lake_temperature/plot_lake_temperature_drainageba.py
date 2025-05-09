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

# Define color ramps
c1=['#045275', '#089099', '#7CCBA2', '#FCDE9C', '#F0746E', '#DC3977', '#7C1D6F']
c2=['#009392', '#39B185', '#9CCB86', '#E9E29C', '#EEB479', '#E88471', '#CF597E']

# Prime plot
fig, ax = plt.subplots(7, 1, figsize=(10,10), sharex=True)
fsize1 = 12
fsize2 = 10
fsize3 = 8
fsize4 = 14
fsty = 'arial'
pad=1.
lloc=4
lw=1.5

df_se1=[]
df_sw1=[]
df_cw1=[]
df_ce1=[]
df_nw1=[]
df_no1=[]
df_ne1=[]
is_dfs=[df_se1, df_sw1, df_cw1, df_ce1, df_nw1, df_no1, df_ne1]

df_se2=[]
df_sw2=[]
df_cw2=[]
df_ce2=[]
df_nw2=[]
df_no2=[]
df_ne2=[]
ic_dfs=[df_se2, df_sw2, df_cw2, df_ce2, df_nw2, df_no2, df_ne2]

# Iterate over lake ST files
for f in list(infile):
    df = pd.read_csv(f, comment='#', index_col=2,
                      na_values=['','nan'], parse_dates=True,
                      sep=',', skip_blank_lines=True) 

    df = df.drop(df[df['ST'] < 0].index)

    # Filter to inventory years
    df['year'] = df.index.year
    df = df.loc[(df['year'] >= 1985) & (df['year'] <= 2023)]
    
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
            ax[b[0]].plot(list(df_y.index.year), list(df_y), c='#B6B6B6', linewidth=0.1)
            
            if 'ICE_SHEET' in list(df_set['d_margin'])[0]:
                is_dfs[b[0]].append(df_y)
            elif 'ICE_CAP' in list(df_set['d_margin'])[0]:
                ic_dfs[b[0]].append(df_y)


for b in range(len(is_dfs)):
    aver_st = pd.concat(is_dfs[b])
    mean = aver_st.resample('YS').mean()
    ax[b].plot(list(mean.index.year), list(mean), marker='o', markersize=3, 
               c=c1[b], linewidth=2, label='Ice Sheet lake average')

    aver_st_ic = pd.concat(ic_dfs[b])
    mean_ic = aver_st_ic.resample('YS').mean()
    ax[b].plot(list(mean_ic.index.year), list(mean_ic), linestyle='--', 
               marker='o', markersize=3, 
               c=c2[b], linewidth=2, label='Ice Cap lake average')

    props1 = dict(boxstyle='round', facecolor=c1[b], alpha=0.2)
    props2 = dict(boxstyle='round', facecolor=c2[b], alpha=0.2)
    ax[b].text(0.03, 0.8, basins[b], fontsize=fsize2, horizontalalignment='center', 
            bbox=props1, transform=ax[b].transAxes)   
    
    ax[b].text(1.12, 0.11, 'Ice Sheet lakes\n\nNo. lakes: '+str(len(is_dfs[b]))
                +'\n1985: '+str(round(list(mean)[0], 2))
                +'\n2023: '+str(round(list(mean)[-1], 2))
                +'\nDifference: '+str(round(list(mean)[-1]-list(mean)[0], 2)), 
                fontsize=fsize2, horizontalalignment='center', 
                bbox=props1, transform=ax[b].transAxes)         

    ax[b].text(1.35, 0.11, 'Ice Cap lakes\n\nNo. lakes: '+str(len(ic_dfs[b]))
                +'\n2016: '+str(round(list(mean_ic)[0], 2))
                +'\n2023: '+str(round(list(mean_ic)[-1], 2))
                +'\nDifference: '+str(round(list(mean_ic)[-1]-list(mean_ic)[0], 2)), 
                fontsize=fsize2, horizontalalignment='center', 
                bbox=props2, transform=ax[b].transAxes)     

    ax[b].legend(loc=3, fontsize=fsize3) #ncol=2)

fig.text(0.05, 0.5, 'Average surface temperature estimate ($^\circ$C)', 
         va='center', rotation='vertical', fontsize=fsize1)

fig.text(0.4, 0.07, 'Year', va='center', fontsize=fsize1)

for a in range(len(ax)):
    ax[a].set_ylim([0, 10])
    ax[a].set_xlim((1985.0, 2023.0))  
    ax[a].set_yticks([0,2,4,6,8,10])

    if a==range(len(ax))[0]:
        ax[a].set_yticklabels(['0','2','4','6','8','10'])        
    else:
        ax[a].set_yticklabels(['0','2','4','6','8',''])
    ax[a].set_xlim((1985.0, 2023.0))  
    
plt.subplots_adjust(wspace=1, hspace=0, left=0.1, right=0.7)  
    
plt.savefig('lake_temp_drainageba.png',dpi=300)
# plt.show()
    
