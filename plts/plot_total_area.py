# -*- coding: utf-8 -*-

import numpy as np
import geopandas as gp
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# %%
f = '/home/pho/python_workspace/GrIML/misc/iml_2016-2023/final/with_lake_temps_100m_buffer_and_centroids/ALL-ESA-GRIML-IML-fv2.gpkg'
geofile = gp.read_file(f)

a = list(geofile['area_sqkm'])
print(a)

sum_a = sum(a)
print(sum_a)


