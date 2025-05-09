#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:45:38 2025

@author: pho
"""
import datetime as dt
import xarray as xr
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
# from adjust_spines import adjust_spines as adj
from matplotlib import rc


infile1 = 'https://thredds.geus.dk/thredds/fileServer/MassBalance/MB_region.nc'
infile2 = 'https://thredds.geus.dk/thredds/fileServer/MassBalance/MB_sector.nc'

regions = ['NW', 'NO', 'NE', 'CE', 'SE', 'SW', 'CW']

MB = xr.open_dataset(infile1)
MB = MB.sel({'time':slice('1986','2099')})

for r in regions:
    ds_region = MB.sel(region=r)
    ds_region["MB_ROI"].plot()
    plt.title(f"Mass Balance (MB) for {r} Region")
    plt.xlabel("Time")
    plt.ylabel("Mass Balance")
    plt.xlim([dt.datetime(2016,1,1), dt.datetime(2024,12,31)])
    plt.grid()
    plt.show()
    

# fig = plt.figure(1, figsize=(2.5,1.5)) # w,h


# for r in MB['region'].values:

#     fig.clf()
#     fig.set_tight_layout(True)
#     ax = fig.add_subplot(111)
#     kw = {'ax':ax, 'legend':False, 'drawstyle':'steps-post'}

#     MB_r = MB.sel({'region':r})

#     MB_r_SMB = MB_r['SMB_ROI'].to_dataframe(name='M2021')\
#                               .resample('AS')\
#                               .sum()\
#                               .rename(columns={'M2021':'SMB'})
#     MB_r_SMB.plot(color='b', linestyle='-', alpha=0.5, **kw)
#     # ax.fill_between(MB_r_SMB.index,
#     #                 MB_r_SMB.values.flatten(),
#     #                 color='b', alpha=0.25, step='post')
    
#     (-1*(MB_r['D_ROI'] + MB_r['BMB_ROI'])).to_dataframe(name='M2021')\
#                         .resample('AS')\
#                         .sum()\
#                         .rename(columns={'M2021':'D + BMB'})\
#                         .plot(color='gray', linestyle='--', **kw)

#     MB_r_MB = MB_r['MB_ROI'].to_dataframe(name='M2021')\
#                             .resample('AS')\
#                             .sum()\
#                             .rename(columns={'M2021':'MB'})
#     MB_r_MB.plot(color='k', linestyle='-', **kw)

#     # MB_r_pos = MB_r_MB.where(MB_r_MB['MB'] > 0, 0)
#     # MB_r_neg = MB_r_MB.where(MB_r_MB['MB'] < 0, 0)
#     # ax.fill_between(MB_r_pos.index, MB_r_pos.values.flatten(), color='r', alpha=0.1, step='post')
#     # ax.fill_between(MB_r_neg.index, MB_r_neg.values.flatten(), color='b', alpha=0.1, step='post')

#     # ax.fill_between(MB_r_MB.index,
#     #                 MB_r_MB.values.flatten(),
#     #                 color='k',
#     #                 alpha=0.25,
#     #                 step='post')

#     # plt.legend(loc='lower left')

#     # ax.set_ylabel("Mass gain [Gt yr$^{-1}$]")
#     ax.set_ylabel("")
#     ax.set_xlabel("")
#     ax.set_xticks(ax.get_xlim())
#     # ax.set_xticklabels(ax.get_xlim())

#     yt = {'NO':50, 'NE':50, 'NW':100, 'CW':100, 'SW':100, 'SE':200, 'CE':100}
#     ax.set_yticks([-yt[r], 0, yt[r]])
    
#     # if r == 'SE':
#         # ax.set_ylabel('Mass gain [Gt yr$^{-1}$]')
#         # ax.set_xlabel('Time [Year]')
#         # ax.set_xticklabels(['1986','2021'])

#     # adj(ax, ['left','bottom'])

#     # ax.grid(b=True, which='major', axis='y', alpha=0.33)

#     # plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
#     # plt.ion()
#     plt.show()
#     # plt.savefig('fig/MB_ts_'+r+'.png', transparent=True, bbox_inches='tight', dpi=300)
