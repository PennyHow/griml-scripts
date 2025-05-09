#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:09:20 2024

@author: pho
"""
import geopandas as gpd
import glob,re,datetime
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Map inventory file locations
gdf_files = '*IML-fv2.shp'

# Map temperature file locations
indir = 'ST_lakes_2016-2023_inventory/'


# Iterate over lake ST files
dfs=[]
for f in list(glob.glob(indir+'*.csv')):
    df = pd.read_csv(f, comment='#', index_col=2,
                      na_values=['','nan'], parse_dates=True,
                      sep=',', skip_blank_lines=True) 

    df = df.drop(df[df['ST'] < 0].index)

    # Filter to inventory years
    df['year'] = df.index.year
    df = df.loc[(df['year'] >= 2016) & (df['year'] <= 2023)]

    # Filter to August months
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
    df['LC'] = [d[d.index('L'):d.index('L')+4] for d in list(df['system:index'])]
    df = df[['lake_id', 'ST', 'ST_count_int', 'ST_max_int', 'ST_min_int', 
             'ST_stddev_int', 'LC', 'year']]
    dfs.append(df)

df_st = pd.concat(dfs)
df_st = df_st.sort_values(by='lake_id')


for f in sorted(list(glob.glob(gdf_files))):
    gdf = gpd.read_file(f)
    year = int(str(Path(f).stem)[0:4])
    
    df_st_y = df_st.loc[(df_st['year'] == year)]
    
    df_st_mean = df_st_y[['lake_id', 'ST', 'ST_max_int', 'ST_stddev_int']] 
    df_st_mean = df_st_mean.groupby('lake_id').mean()
    
    df_st_max = df_st_y[['lake_id', 'ST_max_int']]     
    df_st_max = df_st_max.groupby('lake_id').max()

    df_st_min = df_st_y[['lake_id', 'ST_min_int']]     
    df_st_min = df_st_min.groupby('lake_id').min()
    
    gdf = gdf.merge(df_st_mean, on='lake_id')
    gdf = gdf.merge(df_st_max, on='lake_id')
    gdf = gdf.merge(df_st_min, on='lake_id')
    
    gdf['temp_aver']=gdf['ST']
    gdf['temp_max']=gdf['ST_max_int_y']
    gdf['temp_min']=gdf['ST_min_int']    
    gdf['temp_stdev']=gdf['ST_stddev_int']
    gdf = gdf[['geometry', 'lake_id', 'margin', 'region', 'lake_name', 
               'area_sqkm', 'length_km', 'temp_aver', 'temp_min', 'temp_max', 
               'temp_stdev', 'method', 'source', 'all_src', 'num_src', 
               'certainty', 'start_date', 'end_date', 'verified', 'verif_by', 
               'edited', 'edited_by']]
    
    # Re-format index
    gdf["row_id"] = gdf.index + 1
    gdf.reset_index(drop=True, inplace=True)
    gdf.set_index("row_id", inplace=True)
    
    gdf.to_file(str(year)+'0101-ESA-GRIML-IML-fv1.shp')
                          
