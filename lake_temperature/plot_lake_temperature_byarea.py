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
# iiml = iiml.dissolve(by='LakeID')
iiml["centroid"] = iiml["geometry"].centroid
iiml['area_sqkm'] = iiml['geometry'].area/1000000

area_bins = [0.10,0.20, 0.50, 1., 5., 150.]
l010 = [] 
l020 = [] 
l050 = [] 
l1 = [] 
l5 = [] 
l10 = [] 
l150 = []
bin_labels = [l010, l020, l050, l1, l5, l150]

c1=['#045275', '#089099', '#7CCBA2', '#F0746E', '#DC3977', '#7C1D6F']

# Prime plot
fig, ax = plt.subplots(len(area_bins), 1, figsize=(10,10), sharex=True)
fsize1 = 12
fsize2 = 10
fsize3 = 8
fsize4 = 14
fsty = 'arial'
pad=1.
lloc=4
lw=1.5


# Iterate over lake ST files
for f in list(infile):
    df = pd.read_csv(f, comment='#', index_col=2,
                      na_values=['','nan'], parse_dates=True,
                      sep=',', skip_blank_lines=True) 

    df = df.drop(df[df['ST'] < 0].index)

    # Filter to inventory years
    df['year'] = df.index.year
    df = df.loc[(df['year'] >= 2016) & (df['year'] <= 2023)]
    
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
        for i in list(set(df.lake_id)):
            df_set = df.loc[(df['lake_id'] == i)]
                    
            # Sort by index and re-define satellite description
            df_set = df_set.sort_index()
            df_set['LC'] = [d[d.index('L'):d.index('L')+4] for d in list(df_set['system:index'])]
            
            # Resample to yearly means
            df_y = df_set['ST'].resample(datetime.timedelta(days=365)).mean()
            
            # Define drainage basin
            lake_id = int(list(df_set['lake_id'])[0])
            pt = iiml.loc[iiml.lake_id == lake_id]
            df_set['centroid'] = list(pt['centroid'])[0]
            df_set['d_basin'] = list(pt['region'])[0]
            df_set['area'] = list(pt['area_sqkm'])[0]
            
            # Plot based on basin
            for a in range(len(area_bins)):
                if list(pt['area_sqkm'])[0] <= area_bins[a]:
                    bin_labels[a].append(df_y)
                    ax[a].plot(list(df_y.index.year), list(df_y), c='#B6B6B6', linewidth=0.1)
                    break


for b in range(len(bin_labels)):
    aver_st = pd.concat(bin_labels[b])
    mean = aver_st.resample('YS').mean()

    ax[b].plot(list(mean.index.year), list(mean), marker='o', markersize=3, 
               c=c1[b], linestyle='--', linewidth=2)

    
    labels = [str(round(s,1)) for s in list(mean)]
    ax[b].annotate(labels[0], xy=(list(mean.index.year)[0]+0.08, list(mean)[0]-1.2))
    ax[b].annotate(labels[-1], xy=(list(mean.index.year)[-1]-0.33, list(mean)[-1]-1.2))
    for a in range(len(labels))[1:-1]:
    # for a in range(len(labels))[::3]:
        ax[b].annotate(labels[a], xy=(list(mean.index.year)[a]-0.15, list(mean)[a]-1.2))

    
    props = dict(boxstyle='round', facecolor=c1[b], alpha=0.3)

    ax[b].text(0.03, 0.8, 'Lakes <= '+str(area_bins[b])+' km\u00b2', fontsize=fsize2, 
               horizontalalignment='left', bbox=props, transform=ax[b].transAxes)   
    
    ax[b].text(1.1, 0.4, 'No. lakes: '+str(len(bin_labels[b]))
               +'\n2016: '+str(round(list(mean)[0], 1))
               +'\n2023: '+str(round(list(mean)[-1], 1))
               +'\nDifference: '+str(round(list(mean)[-1]-list(mean)[0], 1)), 
               fontsize=fsize2, horizontalalignment='center', 
               bbox=props, transform=ax[b].transAxes)         

for a in range(len(ax)):
    ax[a].set_ylim([0, 10])
    ax[a].set_yticks([0,2,4,6,8,10])
    if a==range(len(ax))[0]:
        ax[a].set_yticklabels(['0','2','4','6','8','10'])        
    else:
        ax[a].set_yticklabels(['0','2','4','6','8',''])
    ax[a].set_xlim((2016.0, 2023.0))  

fig.text(0.05, 0.5, 'Average surface temperature estimate ($^\circ$C)', 
         va='center', rotation='vertical', fontsize=fsize1)
fig.text(0.45, 0.07, 'Year', va='center', fontsize=fsize1)

lines = [mlines.Line2D([], [], color='k', linestyle='--', label='August average of all lakes'),
         mlines.Line2D([],[],color='#B6B6B6', label='August average of individual lake')]
ax[-1].legend(bbox_to_anchor=(0.5, -0.48), loc='center', ncol=2, handles=lines, 
             fontsize=fsize2)

plt.subplots_adjust(wspace=1, hspace=0, left=0.1, right=0.85)    
plt.savefig('lake_temp_areabins.png',dpi=300)
# plt.show()
    
