# -*- coding: utf-8 -*-

import numpy as np
import geopandas as gp
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# %%
workspace1 = '/home/pho/python_workspace/GrIML/misc/iml_2016-2023/final/fv3_with_merged_auto_classes_and_manual_classes/*01-ESA-GRIML-IML-fv3.gpkg'
out_dir = '/home/pho/python_workspace/GrIML/misc/iml_2016-2023/stats/'

geofiles=[]
aggfiles=[]
for f in list(sorted(glob.glob(workspace1))):
    print(Path(f).stem)
    geofile = gp.read_file(f)
    ag = geofile.dissolve(by='lake_id')
    geofiles.append(geofile)
    aggfiles.append(ag)

# %%
fsize1 = 14
fsize2 = 12
fsize3 = 10
fsize4 = 8
fsty = 'arial'
pad=1.
lloc=1
lw=1    
c1=['#045275', '#089099', '#7CCBA2', '#FCDE9C', '#F0746E', '#DC3977', '#7C1D6F']
c2=['#009392', '#39B185', '#9CCB86', '#E9E29C', '#EEB479', '#E88471', '#CF597E']
b=['NW', 'NO', 'NE', 'CE', 'SE', 'SW', 'CW']
methods = ['ARCTICDEM', 'S1', 'S2']

#--------------------------------

fig = plt.figure(constrained_layout=False, figsize=(10,13))
gs1 = fig.add_gridspec(nrows=3, ncols=1, left=0.08, right=0.9 , top=0.95,
                       bottom=0.54, wspace=0.05, hspace=0.0, height_ratios=[4,1,1])
ax1 = fig.add_subplot(gs1[0, :])
ax5 = fig.add_subplot(gs1[1, :], sharex=ax1)
ax2 = fig.add_subplot(gs1[2, :], sharex=ax1)

gs2 = fig.add_gridspec(nrows=3, ncols=1, left=0.08, right=0.9, top=0.46,
                       bottom=0.05, wspace=0.05, hspace=0.0,height_ratios=[4,1,1])
ax3 = fig.add_subplot(gs2[0, :])
ax6 = fig.add_subplot(gs2[1, :], sharex=ax3)
ax4 = fig.add_subplot(gs2[2, :], sharex=ax3)

is_nw=[]
is_no=[]
is_ne=[]
is_ce=[]
is_se=[]
is_sw=[]
is_cw=[]
ice_sheet_abun = [is_nw, is_no, is_ne, is_ce, is_se, is_sw, is_cw]
ic_nw=[]
ic_no=[]
ic_ne=[]
ic_ce=[]
ic_se=[]
ic_sw=[]
ic_cw=[]
ice_cap_abun = [ic_nw, ic_no, ic_ne, ic_ce, ic_se, ic_sw, ic_cw]
for ag in aggfiles:
    icesheet = ag[ag['margin'] == 'ICE_SHEET']
    icecap = ag[ag['margin'] == 'ICE_CAP']
    for i in range(len(b)):
        ice_sheet_abun[i].append(icesheet['region'].value_counts()[b[i]])
        ice_cap_abun[i].append(icecap['region'].value_counts()[b[i]])

years=list(range(2016,2024, 1))

bottom1=np.zeros(8)
bottom2=np.zeros(8)

for i in range(len(ice_sheet_abun)):
    p1 = ax1.bar(years, ice_sheet_abun[i], 0.5, color=c1[i],  label=b[i], bottom=bottom1)
    p2 = ax3.bar(years, ice_cap_abun[i], 0.5, color=c2[i], label=b[i], bottom=bottom2)

    bottom1 += ice_sheet_abun[i]
    bottom2 += ice_cap_abun[i]

    ax1.bar_label(p1, label_type='center', fontsize=fsize4)
    ax3.bar_label(p2, label_type='center', fontsize=fsize4)
    
print(bottom1)
print(bottom2)


is_nw=[]
is_no=[]
is_ne=[]
is_ce=[]
is_se=[]
is_sw=[]
is_cw=[]
ice_sheet_area = [is_nw, is_no, is_ne, is_ce, is_se, is_sw, is_cw]
ic_nw=[]
ic_no=[]
ic_ne=[]
ic_ce=[]
ic_se=[]
ic_sw=[]
ic_cw=[]
ice_cap_area = [ic_nw, ic_no, ic_ne, ic_ce, ic_se, ic_sw, ic_cw]
is_nw1=[]
is_no1=[]
is_ne1=[]
is_ce1=[]
is_se1=[]
is_sw1=[]
is_cw1=[]
ice_sheet_total = [is_nw1, is_no1, is_ne1, is_ce1, is_se1, is_sw1, is_cw1]
ic_nw1=[]
ic_no1=[]
ic_ne1=[]
ic_ce1=[]
ic_se1=[]
ic_sw1=[]
ic_cw1=[]
ice_cap_total = [ic_nw1, ic_no1, ic_ne1, ic_ce1, ic_se1, ic_sw1, ic_cw1]
for g in geofiles:
    f = g[g['method'] != 'DEM']
    f = f.dissolve(by='lake_id')
    
    i1 = f[f['margin'] == 'ICE_SHEET']
    i2 = f[f['margin'] == 'ICE_CAP']
    
    i1['area_sqkm']=[poly.area/10**6 for poly in list(i1['geometry'])]
    i2['area_sqkm']=[poly.area/10**6 for poly in list(i2['geometry'])]
    
    for i in range(len(b)):
        isheet = i1[i1['region'] == b[i]]
        ice_sheet_area[i].append(np.median(isheet.area_sqkm))
        ice_sheet_total[i].append(sum(isheet.area_sqkm))
        icap = i2[i2['region'] == b[i]]
        ice_cap_area[i].append(np.median(icap.area_sqkm))
        ice_cap_total[i].append(sum(icap.area_sqkm))

for i in range(len(b)):
    print('\nIce Sheet lakes ' + b[i])
    print('Median: ' + str(ice_sheet_area[i]))
    print('Total: ' + str(ice_sheet_total[i]))
    ax2.plot(years, ice_sheet_area[i], c=c1[i], label=b[i])
    ax5.plot(years, ice_sheet_total[i], c=c1[i], label=b[i])
for i in range(len(b)):
    print('\nPGIC lakes ' + b[i])
    print('Median: ' + str(ice_cap_area[i]))
    print('Total: ' + str(ice_cap_total[i]))
    ax4.plot(years, ice_cap_area[i], c=c2[i], label=b[i])
    ax6.plot(years, ice_cap_total[i], c=c2[i], label=b[i])

props = dict(boxstyle='round', facecolor='#6CB0D6', alpha=0.3)
for a in [ax1,ax3]:
    # a.legend(bbox_to_anchor=(1.01,0.5))   
    handles, labels = a.get_legend_handles_labels()
    a.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.01,0.5))

for a in [ax1,ax2,ax3,ax4,ax5,ax6]:
    a.set_axisbelow(True)
    a.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.5)
    a.set_facecolor("#f2f2f2")

ax1.text(0.01, 1.05, 'Ice Sheet lake change', fontsize=fsize1, 
         horizontalalignment='left', bbox=props, transform=ax1.transAxes)
ax3.text(0.01, 1.05, 'Periphery glaciers/ice caps (PGIC) lake change',
         fontsize=fsize1, horizontalalignment='left', bbox=props, transform=ax3.transAxes)

fig.text(0.5, 0.018, 'Year', ha='center', fontsize=fsize2)
fig.text(0.5, 0.51, 'Year', ha='center', fontsize=fsize2)

fig.text(0.02, 0.76, 'Lake abundance', ha='center',
         rotation='vertical', fontsize=fsize2)
fig.text(0.02, 0.27, 'Lake abundance', ha='center',
         rotation='vertical', fontsize=fsize2)

fig.text(0.012, 0.645, 'Total area', ha='center', va='center',
         rotation='vertical', fontsize=fsize2)
fig.text(0.028, 0.645, r'(km$^2$)', ha='center', va='center',
         rotation='vertical', fontsize=fsize2)
fig.text(0.012, 0.155, 'Total area', ha='center', va='center',
         rotation='vertical', fontsize=fsize2)
fig.text(0.028, 0.155, r'(km$^2$)', ha='center', va='center',
         rotation='vertical', fontsize=fsize2)

fig.text(0.012, 0.565, 'Median area', ha='center', va='center',
         rotation='vertical', fontsize=fsize2)
fig.text(0.028, 0.565, r'(km$^2$)', ha='center', va='center',
         rotation='vertical', fontsize=fsize2)
fig.text(0.012, 0.075, 'Median area', ha='center', va='center',
         rotation='vertical', fontsize=fsize2)
fig.text(0.028, 0.075, r'(km$^2$)', ha='center', va='center',
         rotation='vertical', fontsize=fsize2)

fig.text(0.016, 0.96, 'a.', ha='left', fontsize=fsize1+4)
fig.text(0.016, 0.47, 'b.', ha='left', fontsize=fsize1+4)

ax2.set_yticks([0.0,0.25,0.5,0.75])
ax2.set_yticklabels(['0.0','0.25','0.50', ''])
ax4.set_yticks([0.0,0.25,0.5,0.75])
ax4.set_yticklabels(['0.0','0.25','0.50', ''])

ax5.set_yticks([0,300,600,900])
ax5.set_yticklabels(['0','300','600', ''])
ax6.set_yticks([0,100,200,300])
ax6.set_yticklabels(['0','100','200', ''])

# fig.tight_layout(pad=3.0)
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()
plt.savefig(out_dir+'lake_change_by_region.png', dpi=300)


