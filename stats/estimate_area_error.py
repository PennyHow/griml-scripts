#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import geopandas as gpd
import glob


def get_method_area(g, method):
    '''Compute classified lake area from a specific classification method'''
    f = g[g['method'] == method]
    f['idx'] = f['lake_id']
    f = f.dissolve(by='idx')
    f['area_sqkm']=[poly.area/10**6 for poly in list(f['geometry'])]
    return f[['lake_id','region','startdate','area_sqkm']]
    

def estimate_area_error(infile):
    '''Estimate area error by comparing the area of SAR and VIS classified 
    lakes in a inventory, or series of inventories
    
    Parameters
    ----------
    infile : str, list
        File path or list of file paths for estimating abundancy error from
    '''
    # For a series of files
    if type(infile) is list:
        gdfs=[]
        for g in infile:
            
            # Load geodataframe
            print('Loading ' + g)
            gdf = gpd.read_file(g)

            # Get area for SAR and VIS classifications
            sar = get_method_area(gdf, 'SAR')
            vis = get_method_area(gdf, 'VIS')
            
            # Retain common classifications
            common = pd.merge(sar,vis,on=['lake_id'], how='inner')
            common['area_sqkm_diff'] = np.abs(common['area_sqkm_x']-common['area_sqkm_y'])
            gdfs.append(common)
        
        # Concatenate all classifications
        all_common_lakes = pd.concat(gdfs)  

      
    # For a single file
    else:
        gdf = gpd.read_file(infile)
        
        # Get area for SAR and VIS classifications
        sar = get_method_area(gdf, 'SAR')
        vis = get_method_area(gdf, 'VIS')
        
        # Retain common classifications
        all_common_lakes = pd.merge(sar,vis,on=['lake_id'], how='inner')       

    # Get area maximum
    all_common_lakes['area_sqkm_max'] = all_common_lakes[['area_sqkm_x', 'area_sqkm_y']].max(axis=1)

    # Calculate error as percentage
    all_common_lakes['diff_perc'] = (all_common_lakes['area_sqkm_diff']/all_common_lakes['area_sqkm_max'])*100

#    perc = []
#    for i,j in all_common_lakes.iterrows():
#        perc.append((j['area_sqkm_diff']/max([j['area_sqkm_x'],j['area_sqkm_y']]))*100)
#    all_common_lakes['diff_perc']=perc

    return all_common_lakes

if __name__ == "__main__":
    indir = '/home/pho/python_workspace/GrIML/misc/iml_2016-2025/final_v4/*01-ESA-GRIML-IML-fv4.gpkg'
    # indir = '/home/pho/python_workspace/GrIML/misc/iml_2016-2023/final/fv3_with_merged_auto_classes_and_manual_classes/*01-ESA-GRIML-IML-fv3.gpkg'
    infiles = sorted(list(glob.glob(indir)))
    df = estimate_area_error(infiles)
    print(df)
    df.to_csv('/home/pho/Desktop/lake_area_error.csv')

    print('Lake area error estimate for all lakes (' + str(len(df)) + ' lakes)')
    print('Number of lakes: ' + str(len(df)))
    print('Total lake area (SAR): ' + str(round(sum(df['area_sqkm_x']),2)))
    print('Total lake area (VIS): ' + str(round(sum(df['area_sqkm_y']),2)))
    print('Total lake area (max): ' + str(round(sum(df['area_sqkm_max']),2)))

    err = np.abs(sum(df['area_sqkm_x'])-sum(df['area_sqkm_y']))
    print('Total error: ' + str(round(err, 2)))
    print('Total error %: ' + str(round((err / sum(df['area_sqkm_max']))*100, 0)) + ' %')

    print('Average difference: ' + str(round(np.average(df['area_sqkm_diff']), 2)))
    print('Median difference: ' + str(round(np.median(df['area_sqkm_diff']), 2)))
    print('Average % difference: ' + str(round(np.average(df['diff_perc']), 0)))
    print('Median % difference: ' + str(round(np.median(df['diff_perc']), 2)))

    print('\nLake area error estimate by lake size')
    steps = [0.1, 0.2, 0.5, 1.0, 5.0, 150.0]

    print('Lake area error estimate <= ' + str(steps[0]))
    f = df[df['area_sqkm_max'] <= steps[0]]
    print('Number of lakes: ' + str(len(f)))
    print('Total lake area (SAR): ' + str(round(sum(f['area_sqkm_x']),2)))
    print('Total lake area (VIS): ' + str(round(sum(f['area_sqkm_y']),2)))
    err = np.abs(sum(f['area_sqkm_x'])-sum(f['area_sqkm_y']))
    print('Total error: ' + str(round(err, 2)))
    print('Total error %: ' + str(round((err / sum(f['area_sqkm_max']))*100, 0)) + ' %')

    for i in range(len(steps))[1:]:
        print ('\nLake area error estimate <= ' + str(steps[i]))
        f = df[(df['area_sqkm_max'] > steps[i-1]) & (df['area_sqkm_max'] <= steps[i])]
        print('Number of lakes: ' + str(len(f)))
        print('Total lake area (SAR): ' + str(round(sum(f['area_sqkm_x']), 2)))
        print('Total lake area (VIS): ' + str(round(sum(f['area_sqkm_y']), 2)))
        err = np.abs(sum(f['area_sqkm_x']) - sum(f['area_sqkm_y']))
        print('Total error: ' + str(round(err, 2)))
        print('Total error %: ' + str(round((err / sum(f['area_sqkm_max'])) * 100, 0)) + ' %')



    print('\nLake area error estimate by region')
    steps = ['SW','SE','CE','NE','NO','NW','CW']

    for i in steps:
        print ('\nLake area error estimate for ' + str(i))
        f = df[df['region_x'] == i]
        print('Number of lakes: ' + str(len(f)))
        print('Total lake area (SAR): ' + str(round(sum(f['area_sqkm_x']), 2)))
        print('Total lake area (VIS): ' + str(round(sum(f['area_sqkm_y']), 2)))

        err = np.abs(sum(f['area_sqkm_x']) - sum(f['area_sqkm_y']))
        print('Total error: ' + str(round(err, 2)))
        print('Total error %: ' + str(round((err / sum(f['area_sqkm_max'])) * 100, 0)) + ' %')
        print('Max error: '+ str(max(f['area_sqkm_diff'])))
        print('Max error %: '+ str(max(f['diff_perc'])))

    print('\nLake area error estimate by year')
    steps = ['20160701','20170701','20180701','20190701','20200701','20210701','20220701','20230701']

    for i in steps:
        print ('\nLake area error estimate for ' + str(i))
        f = df[df['start_date_x'] == i]
        print('Number of lakes: ' + str(len(f)))
        print('Total lake area (SAR): ' + str(round(sum(f['area_sqkm_x']), 2)))
        print('Total lake area (VIS): ' + str(round(sum(f['area_sqkm_y']), 2)))

        err = np.abs(sum(f['area_sqkm_x']) - sum(f['area_sqkm_y']))
        print('Total error: ' + str(round(err, 2)))
        print('Total error %: ' + str(round((err / sum(f['area_sqkm_max'])) * 100, 0)) + ' %')
