'''
Script for finding intersecting polygons in two files, and then
returning all retained polygons from file #1
'''

import geopandas as gpd

# Load the shapefiles
gdf1 = gpd.read_file('/home/pho/python_workspace/GrIML/misc/iml_2016-2023/merged/2024_merged/2024_merged.shp')
gdf2 = gpd.read_file('/home/pho/python_workspace/GrIML/misc/iml_2016-2023/final/fv3_with_merged_auto_classes_and_manual_classes/ALL-ESA-GRIML-IML-fv3.gpkg')

# Ensure both GeoDataFrames use the same coordinate reference system (CRS)
if gdf1.crs != gdf2.crs:
    gdf2 = gdf2.to_crs(gdf1.crs)

# Perform spatial join: keep only gdf1 polygons that intersect with any in gdf2
gdf1_overlap = gdf1[gdf1.geometry.intersects(gdf2.unary_union)]

# Save to new shapefile if needed
gdf1_overlap.to_file('/home/pho/Desktop/2024_merged_intersects.shp')