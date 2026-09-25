#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:41:24 2024

@author: pho
"""
import geopandas as gpd
import glob, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Centroids of all automatically classified lakes
gdf1 = gpd.read_file(
    '/home/pho/python_workspace/GrIML/misc/iml_2016-2023/final/fv2_with_lake_temps_100m_buffer_and_centroids/ALL-ESA-GRIML-IML-MERGED-centroids-fv2.gpkg'
    )
print(gdf1)

# Load inventory point file with lake_id, region, basin-type and placename info
gdf2 = gpd.read_file(
    "/home/pho/python_workspace/GrIML/misc/iml_2016-2023/manual_validation/CURATED-ESA-GRIML-IML-fv1.shp"
    )
print(gdf2)

gdf1 = gdf1.drop(gdf1[gdf1.geometry == None].index)
gdf2 = gdf2.drop(gdf2[gdf2.geometry == None].index)
print(len(gdf1))
print(len(gdf2))

# 1. Identify the lake_ids present in gdf1
existing_lake_ids = set(gdf1['lake_id'])

# 2. Filter gdf2 to only rows with lake_ids not in gdf1
missing_lake_ids = gdf2[~gdf2['lake_id'].isin(existing_lake_ids)]

# 3. Add 'class_type' to both
gdf1['class_type'] = 'automatic'
missing_lake_ids['class_type'] = 'manual'
missing_lake_ids['verified'] = 'Yes'
missing_lake_ids['verif_by'] = 'How'

# 4. Match column structure: optionally add missing columns from gdf1 to gdf2
for col in gdf1.columns:
    if col not in missing_lake_ids.columns:
        missing_lake_ids[col] = None

# 5. Reorder columns to match gdf1
missing_lake_ids = missing_lake_ids[gdf1.columns]

# 4. Concatenate the two GeoDataFrames
merged_gdf = pd.concat([gdf1, missing_lake_ids], ignore_index=True)

# 5. Ensure the result is a GeoDataFrame
gdf3 = gpd.GeoDataFrame(merged_gdf, geometry='geometry', crs=gdf1.crs)
print(gdf3)

gdf3= gdf3.sort_values(by='lake_id').reset_index(drop=True)
gdf3["idx"] = gdf3['lake_id']

# Reorder columns and index
gdf3 = gdf3[['geometry',
                 'lake_id',
                 'lake_name',
                 'margin',
                 'region',
                 'class_type',
                 'area_sqkm',
                 'length_km',
                 'centroid',
                 'temp_2016',
                 'temp_2017',
                 'temp_2018',
                 'temp_2019',
                 'temp_2020',
                 'temp_2021',
                 'temp_2022',
                 'temp_2023',
                 'temp_all',
                 'verified',
                 'verif_by']]

print(gdf3)


# Save the result
gdf3.to_file(
    "/home/pho/python_workspace/GrIML/misc/iml_2016-2023/final/fv3_with_merged_auto_classes_and_manual_classes/CURATED-ESA-GRIML-IML-fv3.gpkg",
    driver="GPKG")


# Save the result
#filtered_polygons.to_file("cleaned_polygons.shp")
#btree = cKDTree(nB)
#dist, idx = btree.query(nA, k=1)
#gdf2_nearest = gdf2_corr.iloc[idx].drop(columns="geometry").reset_index(drop=True)
#gdf = pd.concat(
#    [
#        gdf1_corr.reset_index(drop=True),
#        gdf2_nearest,
#        pd.Series(dist, name='dist')
#    ],
#    axis=1)

# Reorder columns and index
#gdf_new = gdf[['geometry',
#               'lake_id',
#               'lake_name',
#               'margin',
#               'region',
#               'area_sqkm',
#               'length_km',
#               'temp_aver',
#               'temp_min',
#               'temp_max',
#               'temp_stdev',
#               'method',
#               'source',
#               'all_src',
#               'num_src',
#               'certainty',
#               'start_date',
#               'end_date',
#               'verified',
#               'verif_by',
#               'edited',
#               'edited_by']]
#gdf_new = gdf_new.sort_values(by='lake_id')
    # gdf_new = gdf_new.reset_index(drop=True)

    # # Add sources
    # def _get_indices(mylist, value):
    #     '''Get indices for value in list'''
    #     return[i for i, x in enumerate(mylist) if x==value]

    # col_names=['lake_id', 'source']
    # ids = gdf[col_names[0]].tolist()
    # source = gdf[col_names[1]].tolist()
    # satellites=[]

    # # Construct source list
    # for x in range(len(ids)):
    #     indx = _get_indices(ids, x)
    #     if len(indx) != 0:
    #         res = []
    #         if len(indx) == 1:
    #             res.append(source[indx[0]].split('/')[-1])
    #         else:
    #             unid=[]
    #             for dx in indx:
    #                 unid.append(source[dx].split('/')[-1])
    #             res.append(list(set(unid)))

    #         for z in range(len(indx)):
    #             if len(indx) == 1:
    #                 satellites.append(res)
    #             else:
    #                 satellites.append(res[0])
    #     else:
    #         print(x)
    #         print('Nothing appended!')
    # # Compile lists for appending
    # satellites_names = [', '.join(i) for i in satellites]
    # number = [len(i) for i in satellites]

    # # Return updated geodataframe
    # gdf['all_src']=satellites_names
    # gdf['num_src']=number

    # all_src=[]
    # num_src=[]
    # for idx, i in gdf.iterrows():
    #     idl = i['lake_id']
    #     g = gdf[gdf['lake_id'] == idl]
    #     source = list(set(list(g['source'])))
    #     satellites=''
    #     if len(source)==1:
    #         satellites = satellites.join(source)
    #         num = 1
    #     elif len(source)==2:
    #         satellites = satellites.join(source[0]+', '+source[1])
    #         num = 2
    #     elif len(source)==3:
    #         satellites = satellites.join(source[0]+', '+source[1]+', '+source[2])
    #         num = 3
    #     else:
    #         print('Unknown number of sources detected')
    #         print(source)
    #         satellites=None
    #         num=None
    #     all_src.append(satellites)
    #     num_src.append(num)
    # satellites
    # gdf['all_src']=all_src
    # gdf['num_src']=num_src

    # # Add certainty score
    # def _get_score(value, search_names, scores):
    #     '''Determine score from search string'''
    #     if search_names[0] in value:
    #         return scores[0]
    #     elif search_names[1] in value:
    #         return scores[1]
    #     elif search_names[2] == value:
    #         return scores[2]
    #     else:
    #         return None

    # source='all_src'
    # search_names = ['S1','S2','ARCTICDEM']
    # scores = [0.298, 0.398, 0.304]
    # cert=[]
    # srcs = list(gdf[source])

    # for a in range(len(srcs)):
    #     if srcs[a].split(', ')==1:
    #         out = _get_score(srcs.split(', '))
    #         cert.append(out)
    #     else:
    #         out=[]
    #         for b in srcs[a].split(', '):
    #             out.append(_get_score(b, search_names, scores))
    #         cert.append(sum(out))

    # gdf['certainty'] = cert

    # # Add average summer temperature fields
    # gdf['temp_aver']=''
    # gdf['temp_max']=''
    # gdf['temp_min']=''
    # gdf['temp_stdev']=''
    # gdf['temp_src']=''
    # gdf['temp_num']=''

    # # Add verification and manual intervention fields
    # gdf['verified']='Yes'
    # gdf['verif_by']='How'
    # gdf['edited']=''
    # gdf['edited_by']=''

 #   # Re-format index
 #   gdf_new["row_id"] = gdf_new.index + 1
 #   gdf_new.reset_index(drop=True, inplace=True)
 #   gdf_new.set_index("row_id", inplace=True)

#    print(len(gdf_new))

#    gdf_new.to_file(str(year) + '-ESA-GRIML-IML-fv1.shp')

