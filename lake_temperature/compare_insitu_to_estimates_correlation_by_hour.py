#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:56:05 2024

@author: pho
"""

import glob,re,datetime,math
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import mean_squared_error
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Define files
typ = 'water'
infile1 = 'ee-chart-badeso-'+typ+'.csv'
infile2 = 'ee-chart-qassi-'+typ+'.csv'
infile3 = 'ee-chart-russell-center-'+typ+'.csv'
infile5 = 'ee-chart-st675-'+typ+'.csv'
infile6 = 'ee-chart-st923-'+typ+'.csv'
infile7 = 'ee-chart-st924-'+typ+'.csv'
infile8 = 'ee-chart-lake29a-center-'+typ+'.csv'
infile9 = 'ee-chart-lake35-center-'+typ+'.csv'

insitu_file1 = list(glob.glob('gem_temp_data/*.csv'))[0]
insitu_file2_bade = 'gem_temp_data/gem_recent_lake_data/Temperature_Badeso_2m.csv'
insitu_file2_qassi = 'gem_temp_data/gem_recent_lake_data/qassiso_2017_to_2023.csv'
insitu_file3 = 'russell_lake_kkk_obvs.csv'
insitu_file4 = 'insitu_lake_data_2024.csv'
insitu_file5 = 'lake29a_lake35_insitu_temp.csv'


# Load insitu data
def load_insitu(file1, separator='\t'):
    return pd.read_csv(file1, comment='#',
                       na_values=['','nan', '-9999'], parse_dates=True,
                       sep=separator, skip_blank_lines=True)

df_insitu_temp = load_insitu(insitu_file1)
df_insitu_bade = load_insitu(insitu_file2_bade, ';')
df_insitu_qassi = load_insitu(insitu_file2_qassi, ',')
df_insitu_russell = load_insitu(insitu_file3, ',') 
df_insitu_dop = load_insitu(insitu_file4, ',') 
df_insitu_2935 = load_insitu(insitu_file5,',')

# Reformat GEM insitu data
def reformat_insitu(df_insitu, lake, depth, date_cols=['Date','Time'], depth_col='Depth', date_format='%Y-%m-%d %H:%M'):
    df = df_insitu.loc[(df_insitu['Lake']==lake) & 
                                (df_insitu[depth_col]==depth)]
    if len(date_cols)>1:
        df['datetime'] = pd.to_datetime(df[date_cols[0]]+' '+df[date_cols[1]], format=date_format)
    elif len(date_cols)==1:
        df['datetime'] = pd.to_datetime(df[date_cols[0]], format=date_format)
    df = df.set_index('datetime')
    return df

bade_insitu_2m = reformat_insitu(df_insitu_temp, 'Badesø', 2)
qassi_insitu_2m = reformat_insitu(df_insitu_temp, 'Qassi-sø', 2)
bade_insitu_2m_d = bade_insitu_2m['Temperature'].resample(datetime.timedelta(days=1)).mean()
qassi_insitu_2m_d = qassi_insitu_2m['Temperature'].resample(datetime.timedelta(days=1)).mean()

df_insitu_bade['Temperature'] = df_insitu_bade['temp_2m']
df_insitu_qassi['Temperature'] = df_insitu_qassi['temp_2m']
df_insitu_bade['datetime'] = pd.to_datetime(df_insitu_bade['datetime_utc'], format='%d-%m-%Y %H:%M')
df_insitu_qassi['datetime'] = pd.to_datetime(df_insitu_qassi['datetime_utc'], format='%Y-%m-%d %H:%M:%S')
df_insitu_bade = df_insitu_bade.set_index('datetime')
df_insitu_qassi = df_insitu_qassi.set_index('datetime')
    
bade_insitu_recent_d = df_insitu_bade['Temperature'].resample(datetime.timedelta(days=1)).mean()
qassi_insitu_recent_d = df_insitu_qassi['Temperature'].resample(datetime.timedelta(days=1)).mean()

bade_insitu_2m_all = pd.concat([bade_insitu_2m_d, bade_insitu_recent_d])
qassi_insitu_2m_all = pd.concat([qassi_insitu_2m_d, qassi_insitu_recent_d])

bade_insitu_2m_all = bade_insitu_2m_all.drop_duplicates()
qassi_insitu_2m_all = qassi_insitu_2m_all.drop_duplicates()

# Reformat Russell insitu data
df_insitu_russell['datetime'] = pd.to_datetime(df_insitu_russell['TimeStamp'], format='%Y/%m/%d %H:%M:%S')
df_insitu_russell = df_insitu_russell.set_index('datetime')
df_insitu_russell = df_insitu_russell[df_insitu_russell['WaterLevel']>=0]
df_insitu_russell = df_insitu_russell[df_insitu_russell['WaterLevel']<=200]
russell_insitu_0m_d = df_insitu_russell['Temperature'].resample('60min').mean()

# Reformat DOP insitu data
df_insitu_dop['datetime'] = pd.to_datetime(df_insitu_dop['insitu_timestamp_UTC'], format='%d/%m/%Y %H:%M')
df_insitu_dop = df_insitu_dop.set_index('datetime')
df_insitu_dop['Temperature'] = df_insitu_dop['insitu_wt_2m']

df_insitu_924 = df_insitu_dop[df_insitu_dop['station_id']=='st924']
df_insitu_923 = df_insitu_dop[df_insitu_dop['station_id']=='st923']
df_insitu_675 = df_insitu_dop[df_insitu_dop['station_id']=='st675']

insitu924_2m_d = df_insitu_924['Temperature'].resample('60min').mean()
insitu923_2m_d = df_insitu_923['Temperature'].resample('60min').mean()
insitu675_2m_d = df_insitu_675['Temperature'].resample('60min').mean()


# Reformat Lake 29a/35 insitu data
df_insitu_2935['datetime'] = pd.to_datetime(df_insitu_2935['Date/time'], format='%d/%m/%Y %H:%M')
df_insitu_2935 = df_insitu_2935.set_index('datetime')

df_insitu_29a = df_insitu_2935.copy()
df_insitu_35 = df_insitu_2935.copy()

df_insitu_29a['Temperature'] = df_insitu_29a['Lake 29A water temp']
df_insitu_35['Temperature'] = df_insitu_35['Lake 35 water temp']

df_insitu_29a_d = df_insitu_29a['Temperature'].resample('60min').mean()
df_insitu_35_d = df_insitu_35['Temperature'].resample('60min').mean()

# Load RS estimates
def load_rs_estimates(infile, time_col='system:time_start', time_format='%b %d, %Y'):
    df= pd.read_csv(infile, comment='#',
                    na_values=['','nan'], parse_dates=True,
                    sep=',', skip_blank_lines=True)    
    df["temperature"] = df[["ST_L5", "ST_L7", "ST_L8", "ST_L9"]].sum(axis=1)
    df['datetime']=pd.to_datetime(df[time_col], format=time_format)
    df = df.set_index('datetime')
    df = df.loc[(df['temperature']>=0.0)]
    # df.drop(['system:time_start', 'ST_L5', 'ST_L7', 'ST_L8', 'ST_L9'])   
    return df

bade_rs= load_rs_estimates(infile1)
qassi_rs= load_rs_estimates(infile2)
# bade_rs=bade_rs['temperature'].resample(datetime.timedelta(days=1)).mean()
# qassi_rs=qassi_rs['temperature'].resample(datetime.timedelta(days=1)).mean()
print(bade_rs)
print(qassi_rs)

russell_rs23 = load_rs_estimates(infile3, 'formatted_date', '%Y-%m-%d %H:%M:%S')
# russell_rs24 = load_rs_estimates(infile4, 'formatted_date', '%Y-%m-%d %H:%M:%S')
russell_rs23= russell_rs23['temperature'].resample('60min').mean()
# russell_rs24= russell_rs24['temperature'].resample('60min').mean()

st675_rs = load_rs_estimates(infile5, 'formatted_date', '%Y-%m-%d %H:%M:%S')
st923_rs = load_rs_estimates(infile6, 'formatted_date', '%Y-%m-%d %H:%M:%S')
st924_rs = load_rs_estimates(infile7, 'formatted_date', '%Y-%m-%d %H:%M:%S')
lake29a_rs = load_rs_estimates(infile8, 'formatted_date', '%Y-%m-%d %H:%M:%S')
lake35_rs = load_rs_estimates(infile9, 'formatted_date', '%Y-%m-%d %H:%M:%S')

st675_rs= st675_rs['temperature'].resample('60min').mean()
st923_rs= st923_rs['temperature'].resample('60min').mean()
st924_rs= st924_rs['temperature'].resample('60min').mean()
lake29a_rs= lake29a_rs['temperature'].resample('60min').mean()
lake35_rs= lake35_rs['temperature'].resample('60min').mean()

# Merge in-situ and RS datasets
def merge_datasets(insitu, rs):
    all_obvs= pd.merge(left=insitu, left_on='datetime',right=rs, right_on='datetime')
    all_obvs['temperature'] = all_obvs['temperature'].replace({'0.0':np.nan, 0:np.nan})
    all_obvs = all_obvs[all_obvs['Temperature'].notna()]
    all_obvs = all_obvs[all_obvs['temperature'].notna()]
    return all_obvs

bade_all = merge_datasets(bade_insitu_2m_all, bade_rs)
bade_all = bade_all[bade_all.index != dt.datetime(2014, 8, 15)]
bade_all = bade_all[bade_all.index != dt.datetime(2014, 8, 22)]

qassi_all = merge_datasets(qassi_insitu_2m_all, qassi_rs) 

bade_all_recent = merge_datasets(bade_insitu_recent_d, bade_rs) 
qassi_all_recent = merge_datasets(qassi_insitu_recent_d, qassi_rs) 

russell23_insitu_0m_d = russell_insitu_0m_d[russell_insitu_0m_d.index.year==2023]
russell23_all =  merge_datasets(russell23_insitu_0m_d, russell_rs23) 
russell24_insitu_0m_d = russell_insitu_0m_d[russell_insitu_0m_d.index.year==2024]
# russell24_all =  merge_datasets(russell24_insitu_0m_d, russell_rs24) 
russell24_all =  merge_datasets(russell24_insitu_0m_d, russell_rs23) 
russell24_all = russell24_all[russell24_all['Temperature']>=0]
russell_all = pd.concat([russell23_all, russell24_all])
russell_all = russell_all[russell_all.index.month>=9]

st675_all = merge_datasets(insitu675_2m_d, st675_rs)
st923_all = merge_datasets(insitu923_2m_d, st923_rs)
st924_all = merge_datasets(insitu924_2m_d, st924_rs)

lake29a_all = merge_datasets(df_insitu_29a_d, lake29a_rs)
lake35_all = merge_datasets(df_insitu_35_d, lake35_rs)
lake29a_all = lake29a_all[lake29a_all.index.month<=5]
lake35_all = lake35_all[lake35_all.index.month<=5]

# %%
# Calculate statistics
bade_m, bade_b, bade_r_value, bade_p_value, bade_std_err = scipy.stats.linregress(bade_all['Temperature'], bade_all['temperature'])                                                          
qassi_m, qassi_b, qassi_r_value, qassi_p_value, qassi_std_err = scipy.stats.linregress(qassi_all['Temperature'], qassi_all['temperature'])                                                          
russell_m, russell_b, russell_r_value, russell_p_value, russell_std_err = scipy.stats.linregress(russell_all['Temperature'], russell_all['temperature'])                                                          
st675_m, st675_b, st675_r_value, st675_p_value, st675_std_err = scipy.stats.linregress(st675_all['Temperature'], st675_all['temperature']) 
st923_m, st923_b, st923_r_value, st923_p_value, st923_std_err = scipy.stats.linregress(st923_all['Temperature'], st923_all['temperature'])
st924_m, st924_b, st924_r_value, st924_p_value, st924_std_err = scipy.stats.linregress(st924_all['Temperature'], st924_all['temperature'])
lake29a_m, lake29a_b, lake29a_r_value, lake29a_p_value, lake29a_std_err = scipy.stats.linregress(lake29a_all['Temperature'], lake29a_all['temperature'])
lake35_m, lake35_b, lake35_r_value, lake35_p_value, lake35_std_err = scipy.stats.linregress(lake35_all['Temperature'], lake35_all['temperature'])

bade_mse = mean_squared_error(bade_all['Temperature'], bade_all['temperature'])
qassi_mse = mean_squared_error(qassi_all['Temperature'], qassi_all['temperature'])
russell_mse = mean_squared_error(russell_all['Temperature'], russell_all['temperature'])
st675_mse = mean_squared_error(st675_all['Temperature'], st675_all['temperature'])
st923_mse = mean_squared_error(st923_all['Temperature'], st923_all['temperature'])
st924_mse = mean_squared_error(st924_all['Temperature'], st924_all['temperature'])
lake29a_mse = mean_squared_error(lake29a_all['Temperature'], lake29a_all['temperature'])
lake35_mse = mean_squared_error(lake35_all['Temperature'], lake35_all['temperature'])

# Raise the mean squared error to the power of 0.5 
bade_rmse = (bade_mse)**(1/2)
qassi_rmse = (qassi_mse)**(1/2)
russell_rmse = (russell_mse)**(1/2)
st675_rmse = (st675_mse)**(1/2)
st923_rmse = (st923_mse)**(1/2)
st924_rmse = (st924_mse)**(1/2)
lake29a_rmse = (lake29a_mse)**(1/2)
lake35_rmse = (lake35_mse)**(1/2)

# Print the RMSE
print("The calculated Root Mean Square Error (RMSE) for Badeso is: " + str(bade_rmse))
print("The calculated Root Mean Square Error (RMSE) for Qassi-so is: " + str(qassi_rmse))
print("The calculated Root Mean Square Error (RMSE) for Russell is: " + str(russell_rmse))
print("The calculated Root Mean Square Error (RMSE) for ST675 is: " + str(st675_rmse))
print("The calculated Root Mean Square Error (RMSE) for ST923 is: " + str(st923_rmse))
print("The calculated Root Mean Square Error (RMSE) for ST924 is: " + str(st924_rmse))
print("The calculated Root Mean Square Error (RMSE) for ST924 is: " + str(lake29a_rmse))
print("The calculated Root Mean Square Error (RMSE) for ST924 is: " + str(lake35_rmse))

# Add difference info
def add_difference(df):
    df['diff'] = df['Temperature'] - df['temperature']
    df['diff_perc'] = (df['diff']/df['temperature'])*100
    df['diff'] = df['diff'].abs()
    return df

bade_all = add_difference(bade_all)
qassi_all = add_difference(qassi_all)
russell_all = add_difference(russell_all)
st675_all = add_difference(st675_all)
st923_all = add_difference(st923_all)
st924_all = add_difference(st924_all)
lake29a_all = add_difference(lake29a_all)
lake35_all = add_difference(lake35_all)

# %%
bade_all['month'] = bade_all.index.month
qassi_all['month'] = qassi_all.index.month
russell_all['month'] = russell_all.index.month
st675_all['month'] = st675_all.index.month
st923_all['month'] = st923_all.index.month
st924_all['month'] = st924_all.index.month
lake29a_all['month'] = lake29a_all.index.month
lake35_all['month'] = lake35_all.index.month

dop_insitu = list(st675_all['Temperature'])+list(st923_all['Temperature'])+list(st924_all['Temperature'])
dop_rs = list(st675_all['temperature'])+list(st923_all['temperature'])+list(st924_all['temperature'])
dop_mth = list(st675_all['month'])+list(st923_all['month'])+list(st924_all['month'])
                                                         
dop_m, dop_b, dop_r_value, dop_p_value, dop_std_err = scipy.stats.linregress(dop_insitu, dop_rs)                                                          
dop_mse = mean_squared_error(dop_insitu, dop_rs)

all_insitu = list(bade_all['Temperature'])+list(qassi_all['Temperature'])+list(russell_all['Temperature'])+list(st675_all['Temperature'])+list(st923_all['Temperature'])+list(st924_all['Temperature'])#+list(lake29a_all['Temperature'])+list(lake35_all['Temperature'])
all_rs = list(bade_all['temperature'])+list(qassi_all['temperature'])+list(russell_all['temperature'])+list(st675_all['temperature'])+list(st923_all['temperature'])+list(st924_all['temperature'])#+list(lake29a_all['temperature'])+list(lake35_all['temperature'])
all_mth = list(bade_all['month'])+list(qassi_all['month'])+list(russell_all['month'])+list(st675_all['month'])+list(st923_all['month'])+list(st924_all['month'])#+list(lake29a_all['month'])+list(lake35_all['month'])
all_diff = list(bade_all['diff'])+list(qassi_all['diff'])+list(russell_all['diff'])+list(st675_all['diff'])+list(st923_all['diff'])+list(st924_all['diff'])#+list(lake29a_all['diff'])+list(lake35_all['diff'])
print('Error estimate for all measurements based on difference is: ' +str(np.mean(all_diff)))
                                                         
all_m, all_b, all_r_value, all_p_value, all_std_err = scipy.stats.linregress(all_insitu, all_rs)                                                          
all_mse = mean_squared_error(all_insitu, all_rs)

# Raise the mean squared error to the power of 0.5 
all_rmse = (all_mse)**(1/2)

# Print the RMSE
print("The calculated Root Mean Square Error (RMSE) for all is: " + str(all_rmse))
print("Estimated uncertainty based on average value difference is: " + str(np.mean(all_diff)))

# Prime plot
fig, ((ax0,ax1),(ax2,ax3),(ax4,ax5)) = plt.subplots(3, 2, figsize=(10,15))
fsize1 = 16
fsize2 = 14
fsize3 = 12
fsize4 = 14
fsty = 'arial'
pad=1.
lloc=4
lw=1.5

# Define a colormap and normalize the months
cmap = cm.get_cmap('rainbow', 12)  # Use a colormap with 12 distinct colors
norm = mcolors.Normalize(vmin=1, vmax=12)  # Normalize months (1 to 12)

# For each subplot:
ax0.scatter(list(bade_all['Temperature']), list(bade_all['temperature']),
            marker='o', s=50, c=list(bade_all['month']), cmap=cmap, norm=norm,
            edgecolor='#46494f')
ax1.scatter(list(qassi_all['Temperature']), list(qassi_all['temperature']),
            marker='^', s=50, c=list(qassi_all['month']), cmap=cmap, norm=norm,
            edgecolor='#46494f')
ax2.scatter(list(russell_all['Temperature']), list(russell_all['temperature']),
            marker='s', s=50, c=list(russell_all['month']), cmap=cmap, norm=norm,
            edgecolor='#46494f')
ax3.scatter(list(st675_all['Temperature']), list(st675_all['temperature']),
            marker='X', s=50, c=list(st675_all['month']), cmap=cmap, norm=norm,
            edgecolor='#46494f', label='ST675')
ax3.scatter(list(st923_all['Temperature']), list(st923_all['temperature']),
            marker='*', s=50, c=list(st923_all['month']), cmap=cmap, norm=norm,
            edgecolor='#46494f', label='ST923')
ax3.scatter(list(st924_all['Temperature']), list(st924_all['temperature']),
            marker='D', s=50, c=list(st924_all['month']), cmap=cmap, norm=norm,
            edgecolor='#46494f', label='ST924')
ax3.legend(loc=3)
for i,sh in zip([bade_all, qassi_all, russell_all, st675_all, st923_all, st924_all],
               ['o','^','s','X', '*', 'D']):    
    ax4.scatter(list(i['Temperature']), list(i['temperature']),
                marker=sh, s=50, c=list(i['month']), cmap=cmap, norm=norm,
                edgecolor='#46494f')

props = dict(boxstyle='round', facecolor='#3F8BD7', alpha=0.2)
title=['Qassi-sø','Russell Lake','ST675 / ST923 / ST924','All']
ax0.text(0.05, 0.83, 'Kangerluarsunnguup\nTasia', fontsize=fsize2, horizontalalignment='left', 
            bbox=props, transform=ax0.transAxes)
for a,t in zip([ax1,ax2,ax3,ax4],title):    
    a.text(0.05, 0.9, t, fontsize=fsize2, horizontalalignment='left', 
                bbox=props, transform=a.transAxes)  

for a in [ax0,ax1,ax2,ax3,ax4]:
    a.grid(True)    
    a.set_axisbelow(True)
    a.yaxis.grid(color='gray', linestyle='dashed')
    a.xaxis.grid(color='gray', linestyle='dashed')
    a.set_xlabel('In situ temperature (2 m)', fontsize=fsize2)
    a.set_ylabel('Remotely sensed temperature estimate', fontsize=fsize2)
    a.set_xlim((0,16))
    a.set_ylim((0,16))
    a.set_xticks([0,2,4,6,8,10,12,14,16])    
    a.set_yticks([0,2,4,6,8,10,12,14,16])  

ax2.set_xlabel('In situ temperature (<=2 m)', fontsize=fsize2)
ax2.set_xlim(0,8)
ax2.set_ylim(0,8)
ax2.set_xticks([0,1,2,3,4,5,6,7,8])    
ax2.set_yticks([0,1,2,3,4,5,6,7,8])  
# ax3.set_xlim(6,12)
# ax3.set_ylim(6,12)
# ax3.set_xticks([6,7,8,9,10,11,12])    
# ax3.set_yticks([6,7,8,9,10,11,12])  

ax4.set_xlabel('In situ temperature', fontsize=fsize2)
    
# Add a colorbar to the figure
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Optional: dummy array for colorbar
cbar = fig.colorbar(sm, ax=[ax0, ax1, ax2, ax3, ax4], orientation='vertical')
cbar.ax.tick_params(labelsize=fsize2)
cbar.set_label(label='Month',size=fsize2, rotation=270, labelpad=10)

def add_r_value(ax, m, b, r_value, neg=True):
    if neg==True:
        ax.text(0.4,0.2, f'y = {m:.2f}x - {-b:.2f}\n' +
                   'r\u00b2 =' + str("{:.2f}".format(r_value**2)),
                   fontsize=fsize3, horizontalalignment='left', 
                   bbox=props, transform=ax.transAxes)
    else:
        ax.text(0.4,0.2, f'y = {m:.2f}x + {b:.2f}\n' +
                   'r\u00b2 = ' + str("{:.2f}".format(r_value**2)),
                   fontsize=fsize3, horizontalalignment='left', 
                   bbox=props, transform=ax.transAxes)

add_r_value(ax0, bade_m, bade_b, bade_r_value)
add_r_value(ax1, qassi_m, qassi_b, qassi_r_value)
add_r_value(ax2, russell_m, russell_b, russell_r_value, neg=False)
add_r_value(ax3, dop_m, dop_b, dop_r_value)
add_r_value(ax4, all_m, all_b, all_r_value, neg=False)


ax0.text(0, 1.02, 'a', fontsize=fsize1, horizontalalignment='left', 
          transform=ax0.transAxes)  
ax1.text(0, 1.02, 'b', fontsize=fsize1, horizontalalignment='left', 
          transform=ax1.transAxes)  
ax2.text(0, 1.02, 'c', fontsize=fsize1, horizontalalignment='left', 
          transform=ax2.transAxes)  
ax3.text(0, 1.02, 'd', fontsize=fsize1, horizontalalignment='left', 
          transform=ax3.transAxes)  
ax4.text(0, 1.02, 'e', fontsize=fsize1, horizontalalignment='left', 
          transform=ax4.transAxes) 

fig.delaxes(ax5)
plt.savefig('insitu_validation_correlation_morepts_'+typ+'.png', dpi=300)
# plt.show()

# %%
# Prime plot
fig, ax = plt.subplots(1, figsize=(5,5))
fsize1 = 14
fsize2 = 12
fsize3 = 8
fsize4 = 8
fsty = 'arial'
pad=1.
lloc=4
lw=1.5

# Define a colormap and normalize the months
cmap = cm.get_cmap('rainbow', 12)  # Use a colormap with 12 distinct colors
norm = mcolors.Normalize(vmin=1, vmax=12)  # Normalize months (1 to 12)

# For each subplot:
ax.scatter(list(bade_all['Temperature']), list(bade_all['temperature']),
            marker='o', s=50, c=list(bade_all['month']), cmap=cmap, norm=norm,
            edgecolor='#46494f', label='Kangerluarsunnguup\nTasia')
ax.scatter(list(qassi_all['Temperature']), list(qassi_all['temperature']),
            marker='^', s=50, c=list(qassi_all['month']), cmap=cmap, norm=norm,
            edgecolor='#46494f', label='Qassi-sø (2011-21)')
ax.scatter(list(russell_all['Temperature']), list(russell_all['temperature']),
            marker='s', s=50, c=list(russell_all['month']), cmap=cmap, norm=norm,
            edgecolor='#46494f', label='Russell Lake')
ax.scatter(list(st675_all['Temperature']), list(st675_all['temperature']),
            marker='X', s=50, c=list(st675_all['month']), cmap=cmap, norm=norm,
            edgecolor='#46494f', label='Qassi-sø (2024)') # label='ST675')
ax.scatter(list(st923_all['Temperature']), list(st923_all['temperature']),
            marker='*', s=50, c=list(st923_all['month']), cmap=cmap, norm=norm,
            edgecolor='#46494f', label='Qamanersuaq') # label='ST923')
ax.scatter(list(st924_all['Temperature']), list(st924_all['temperature']),
            marker='D', s=50, c=list(st924_all['month']), cmap=cmap, norm=norm,
            edgecolor='#46494f', label='Asiaq ST924')
# ax.scatter(list(lake29a_all['Temperature']), list(lake29a_all['temperature']),
#             marker='v', s=50, c=list(lake29a_all['month']), cmap=cmap, norm=norm,
#             edgecolor='#46494f', label='Lake 29a') # label='Lake 29a')
# ax.scatter(list(lake35_all['Temperature']), list(lake35_all['temperature']),
#             marker='p', s=50, c=list(lake35_all['month']), cmap=cmap, norm=norm,
#             edgecolor='#46494f', label='Lake 35') # label='Lake 35')
ax.legend(bbox_to_anchor=(1.25, -0.19), ncol=3, fontsize=fsize4)

print(bade_all)
print(qassi_all)
print(russell_all)
print(st675_all)
print(st923_all)
print(st924_all)

ax.grid(True)    
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.xaxis.grid(color='gray', linestyle='dashed')
ax.set_xlabel('In situ surface (<=2 m) temperature ($^\circ$C)', fontsize=fsize2)
ax.set_ylabel(r'Remotely sensed temperature estimate ($^\circ$C)', fontsize=fsize2)
ax.set_xlim((0,16))
ax.set_ylim((0,16))
ax.set_xticks([0,2,4,6,8,10,12,14,16])    
ax.set_yticks([0,2,4,6,8,10,12,14,16])  
    
# Add a colorbar to the figure
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Optional: dummy array for colorbar
cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
cbar.ax.tick_params(labelsize=fsize2)
cbar.set_label(label='Month of observation',size=fsize2, rotation=270, labelpad=15)

#ax.text(0.5,0.07, f'y = {all_m:.2f}x + {all_b:.2f}\n' +
#        'r\u00b2 = ' + str("{:.2f}".format(all_r_value**2)),
#        fontsize=fsize3, horizontalalignment='left',
#        bbox=props, transform=ax.transAxes)

# Compute mean of in situ temperatures
x_mean = np.mean(all_insitu)

# Compute standard error of the intercept
se_b = all_std_err * np.sqrt(np.sum((all_insitu - x_mean) ** 2) / len(all_insitu))

# Modify the regression equation annotation to include errors in both terms
ax.text(0.35, 0.08,
    f'y = ({all_m:.2f} ± {all_std_err:.2f})x + ({all_b:.2f} ± {se_b:.2f})\n' +
    'r\u00b2 = ' + str("{:.2f}".format(all_r_value**2)),
    fontsize=fsize3, horizontalalignment='left',
    bbox=props, transform=ax.transAxes)

plt.subplots_adjust(top=0.92, bottom=0.3, left=0.15, right=0.95)
#plt.savefig('insitu_validation_correlation_all_'+typ+'.png', dpi=300)
plt.show()
