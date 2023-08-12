#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# In this file we draw figures and compute stats. Juries are generated in
# file juryConstruction.py
#
#%% Module import (run before running any cell)

import gzip,pickle,shutil
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import labellines

# set some default colors and sizes
STRcolor = 'Olive'
SARcolor = 'DarkOrange'
RANcolor = 'SaddleBrown'
altcolor = 'Sienna'
edgecolor = 'White'
alt2color = 'GoldenRod'
barcolors = [SARcolor, STRcolor]
barcolor2 = [SARcolor, STRcolor]

publishedFigures = False

if publishedFigures == True:

    # JLaw&Econ requirement
    STRcolor = SARcolor = RANcolor = edgecolor = alt2color = 'Black'
    barcolors = barcolor2 = ['White']*2
    altcolor =  'White'

figsize = 4.5
figsize_single = 3.5
labelfont = 9
xylabelfont = 11

imagedir = '../Exhibits/'
outputdir = '../Simulations/'

#to make sure LaTeX is found
import os
os.environ['PATH'] += '/Library/TeX/texbin'

# checks if your system does not have or cannot read a LaTeX distribution
# and define labels accordingly
if shutil.which('latex'):
    import matplotlib
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]})
    underline_c_label = r'\underline{c}'
    rep_label = r'\textit{REP}'
    ran_label = r'\textit{RAN}'
    str_label = r'\textit{STR}'
    r10 = r'\parbox{5em}{\textit{REP}, \newline $r=.10$}'
    r25 = r'\textit{REP}, $r=.25$'
    r50 = r'\textit{REP}, $r=.50$'
    r75 = r'\textit{REP}, $r=.75$'
    r90 = r'\textit{REP}, $r=.90$'
else:
    underline_c_label = '_c_'
    rep_label = 'REP'
    ran_label = 'RAN'
    str_label = 'STR'
    r10 = 'REP, \n r=.10'
    r25 = 'REP, r=.25'
    r50 = 'REP, r=.50'
    r75 = 'REP, r=.75'
    r90 = 'REP, r=.90'

table_pre = '\n\\begin{tabular}{lccccccc}'
table_pre+= 	'\n	 Polarization &\\multicolumn{2}{c}{Extreme}'
table_pre+= 	'\n 	 &\\multicolumn{2}{c}{Moderate}'
table_pre+= 	'\n 	 &\\multicolumn{2}{c}{Mild}'
table_pre+= 	'\n 	 & (All) \\\\'
table_pre+= 	'\n 	 Procedure    & \\SAR & \\STR  & \\SAR & \\STR  & \\SAR & \\STR  & \\RAN \\\\'
table_pre+= 	'\n 	 \\hline\n'
table_post = '\n\\end{tabular}\n'

# needed for debugging since I run this script from different folders
import os
if os.getcwd()[-7:] == 'Package':
    os.chdir(os.getcwd() + '/Code')

#%% Plot beta densities (fig:betaPDFs)

from math import gamma

x = np.linspace(0,1,100)
alpha = 1
beta = 5
ya_x = np.multiply(x**(alpha-1),(1-x)**(beta-1)) * gamma(alpha+beta)/(gamma(alpha)*gamma(beta))

alpha = 5
beta = 1
yb_x = np.multiply(x**(alpha-1),(1-x)**(beta-1)) * gamma(alpha+beta)/(gamma(alpha)*gamma(beta))

alpha = 2
beta = 4
ya_d = np.multiply(x**(alpha-1),(1-x)**(beta-1)) * gamma(alpha+beta)/(gamma(alpha)*gamma(beta))
alpha = 4
beta = 2
yb_d = np.multiply(x**(alpha-1),(1-x)**(beta-1)) * gamma(alpha+beta)/(gamma(alpha)*gamma(beta))

alpha = 3
beta = 4
ya_m = np.multiply(x**(alpha-1),(1-x)**(beta-1)) * gamma(alpha+beta)/(gamma(alpha)*gamma(beta))
alpha = 4
beta = 3
yb_m = np.multiply(x**(alpha-1),(1-x)**(beta-1)) * gamma(alpha+beta)/(gamma(alpha)*gamma(beta))

fig,(axx,axd,axm) = plt.subplots(1, 3, figsize=(figsize,figsize*1/2+.2), constrained_layout=True)

for ax in [axx,axd,axm]: 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(-0.01,1.01)
    
axd.set_xlabel('Conviction probability')

axx.plot(x,ya_x,color=STRcolor,label='$f_a$')
axx.plot(x,yb_x,'--',color=SARcolor,label='$f_b$')
axx.set_title('(a) Extreme \n\n $f_a(c): Beta(1,5)$ \n $f_b(c): Beta(5,1)$', fontsize=xylabelfont)
labellines.labelLines(axx.get_lines(), align=False, fontsize=xylabelfont, xvals=[.18,.82],bbox={'alpha': 0}, )

axd.plot(x,ya_d,color=STRcolor,label='$f_a$')
axd.plot(x,yb_d,'--',color=SARcolor,label='$f_b$')
axd.set_title('(b) Moderate \n\n $f_a(c): Beta(2,4)$ \n $f_b(c): Beta(4,2)$', fontsize=xylabelfont)
labellines.labelLines(axd.get_lines(), align=False, fontsize=xylabelfont, xvals=[.14,.86],bbox={'alpha': 0}, )

axm.plot(x,ya_m,color=STRcolor,label='$f_a$')
axm.plot(x,yb_m,'--',color=SARcolor,label='$f_b$')
axm.set_title('(c) Mild \n\n $f_a(c): Beta(3,4)$ \n $f_b(c): Beta(4,3)$', fontsize=xylabelfont)
labellines.labelLines(axm.get_lines(), align=False, fontsize=xylabelfont, xvals=[.2,.8],bbox={'alpha': 0}, )

fig.tight_layout()
plt.savefig(imagedir+'betaPDFs.pdf')

#%% fig:atleast1-3betas (for proposition 1)

def new_func(datafile):
    fname = gzip.open(outputdir+datafile,'rb')
    resultuni = pickle.load(fname)
    baseargs = resultuni['baseargs']
    njuries = len(resultuni['juriestSAR'][:,0])
    fname.close()

    xSAR = resultuni['juriesxSAR']
    xSTR = resultuni['juriesxSTR']
    xRAN = resultuni['juriesxRAN']

    nLess_c_SAR = np.array([])
    nLess_c_STR = np.array([])
    nLess_c_RAN = np.array([])
    nMore_c_SAR = np.array([])
    nMore_c_STR = np.array([])
    nMore_c_RAN = np.array([])

    xx = np.arange(0, 1.01, 0.01)
    for c in xx:
        nLess_c_SAR = np.append(nLess_c_SAR, np.sum(np.count_nonzero(xSAR<c, axis=1) >=1 ) /njuries)
        nLess_c_STR = np.append(nLess_c_STR, np.sum(np.count_nonzero(xSTR<c, axis=1) >=1 ) /njuries)
        nLess_c_RAN = np.append(nLess_c_RAN, np.sum(np.count_nonzero(xRAN<c, axis=1) >=1 ) /njuries)
        nMore_c_SAR = np.append(nMore_c_SAR, np.sum(np.count_nonzero(xSAR>c, axis=1) >=1 ) /njuries)
        nMore_c_STR = np.append(nMore_c_STR, np.sum(np.count_nonzero(xSTR>c, axis=1) >=1 ) /njuries)
        nMore_c_RAN = np.append(nMore_c_RAN, np.sum(np.count_nonzero(xRAN>c, axis=1) >=1 ) /njuries)
    return xRAN,nLess_c_SAR,nLess_c_STR,nLess_c_RAN,xx

## extreme polarization
xRAN_x, nLess_c_SAR_x, nLess_c_STR_x, nLess_c_RAN_x, xx = new_func('beta-1-5-12j-75pcT1.pickle.gz')
xRAN_d, nLess_c_SAR_d, nLess_c_STR_d, nLess_c_RAN_d, xx = new_func('beta-2-4-12j-75pcT1.pickle.gz')
xRAN_m, nLess_c_SAR_m, nLess_c_STR_m, nLess_c_RAN_m, xx = new_func('beta-3-4-12j-75pcT1.pickle.gz')

fig,(axx,axd,axm) = plt.subplots(1,3,figsize=(figsize,figsize*7/12), sharey=True)
for ax in [axx,axd,axm]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlim((0,.45))
    ax.set_xlabel('Threshold '+underline_c_label,fontsize=xylabelfont)
    ax.set_xticks([0,0.2,0.4])

axx.set_ylabel('Fraction of juries',fontsize=xylabelfont)

axx.plot(xx, nLess_c_RAN_x, ':', color=RANcolor, linewidth=2, label=ran_label)
axx.plot(xx, nLess_c_SAR_x, '', color=SARcolor, label=rep_label)
axx.plot(xx, nLess_c_STR_x, '--', color=STRcolor, label=str_label)
axd.plot(xx, nLess_c_RAN_d, ':', color=RANcolor, linewidth=2, label=ran_label)
axd.plot(xx, nLess_c_SAR_d, '', color=SARcolor, label=rep_label)
axd.plot(xx, nLess_c_STR_d, '--', color=STRcolor, label=str_label)
axm.plot(xx, nLess_c_RAN_m, ':', color=RANcolor, linewidth=2, label=ran_label)
axm.plot(xx, nLess_c_SAR_m, '', color=SARcolor, label=rep_label)
axm.plot(xx, nLess_c_STR_m, '--', color=STRcolor, label=str_label)

#ax.legend(loc="lower right")
labellines.labelLines(axx.get_lines(), align=False, fontsize=labelfont,
                      xvals = [.1,.25,.35],
                      yoffsets=[.18,.07, -.15],
                      bbox={'alpha': 0},
                            )
labellines.labelLines(axd.get_lines(), align=False, fontsize=labelfont,
                      xvals = [.15,.24,.37],
                      yoffsets=[.29,0, -.18],
                      bbox={'alpha': 0},
                            )
labellines.labelLines(axm.get_lines(), align=False, fontsize=labelfont,
                      xvals = [.22,.25,.37],
                      yoffsets=[.37,0, -.1],
                      bbox={'alpha': 0},
                            )

axx.set_title('(a) Extreme', fontsize=xylabelfont)
axd.set_title('(b) Moderate', fontsize=xylabelfont)
axm.set_title('(c) Mild', fontsize=xylabelfont)
fig.tight_layout()
plt.savefig(imagedir+'prop1-beta-all.pdf')


#%%
# Numbers in text
textstring = ''

# extreme polarization
textstring += 'Extreme, 10th pctile='+str(np.percentile(xRAN_x,10))
textstring += '\n verify: '+str(xx[10:11])
textstring += '\n STR' +str(nLess_c_STR_x[10:11])
textstring += '\n SAR' +str(nLess_c_SAR_x[10:11])
textstring += '\n RAN' +str(nLess_c_RAN_x[10:11])

# moderate polarization
textstring += '\n\nModerate, 10th pctile' + str(np.percentile(xRAN_d,10))
textstring += '\nverify: '+str(xx[24:25])
textstring += '\n STR ' +str(nLess_c_STR_d[24:25])
textstring += '\n SAR ' +str(nLess_c_SAR_d[24:25])
textstring += '\n RAN ' +str(nLess_c_RAN_d[24:25])

# mild polarization
textstring += '\n\nMild 10th pctile=' + str(np.percentile(xRAN_m,10))
textstring += '\nverify: '+str(xx[28:29])
textstring += '\n STR' + str(nLess_c_STR_m[28:29])
textstring += '\n SAR' + str(nLess_c_SAR_m[28:29])
textstring += '\n RAN' + str(nLess_c_RAN_m[28:29])

print(textstring)

with open(imagedir + 'figuresintext.txt', 'w') as outputfile:
    outputfile.write(textstring)
outputfile.close()

#%% Proposition 2 counterexample (fig:prop2-uni)

name = gzip.open(outputdir+'uni-0.1-0.9-1j-25pcT1.pickle.gz','rb')

resultuni = pickle.load(name)
baseargs = resultuni['baseargs']
njuries = len(resultuni['juriestSAR'][:,0])
name.close()

xSAR = resultuni['juriesxSAR']
xSTR = resultuni['juriesxSTR']
xRAN = resultuni['juriesxRAN']
nLess_c_SAR = np.array([])
nLess_c_STR = np.array([])
nLess_c_RAN = np.array([])
nMore_c_SAR = np.array([])
nMore_c_STR = np.array([])
nMore_c_RAN = np.array([])

xx = np.arange(0, 1.01, 0.01)
for c in xx:
    nLess_c_SAR = np.append(nLess_c_SAR, np.sum(np.count_nonzero(xSAR<c, axis=1) >=1 ) /njuries)
    nLess_c_STR = np.append(nLess_c_STR, np.sum(np.count_nonzero(xSTR<c, axis=1) >=1 ) /njuries)
    nLess_c_RAN = np.append(nLess_c_RAN, np.sum(np.count_nonzero(xRAN<c, axis=1) >=1 ) /njuries)
    nMore_c_SAR = np.append(nMore_c_SAR, np.sum(np.count_nonzero(xSAR>c, axis=1) >=1 ) /njuries)
    nMore_c_STR = np.append(nMore_c_STR, np.sum(np.count_nonzero(xSTR>c, axis=1) >=1 ) /njuries)
    nMore_c_RAN = np.append(nMore_c_RAN, np.sum(np.count_nonzero(xRAN>c, axis=1) >=1 ) /njuries)

fig,(ax) = plt.subplots(1,1,figsize=(figsize_single,figsize_single*3/4))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlim((0,0.1))
ax.set_ylim((0,0.9))
ax.set_xlabel('Threshold '+underline_c_label,fontsize=xylabelfont)
ax.set_ylabel('Fraction of juries',fontsize=xylabelfont)

ax.plot(xx, nLess_c_RAN, ':', color=RANcolor, linewidth=2, label=ran_label)
ax.plot(xx, nLess_c_SAR, '', color=SARcolor, label=rep_label)
ax.plot(xx, nLess_c_STR, '--', color=STRcolor, label=str_label)
#ax.legend(loc="center left")
labellines.labelLines(plt.gca().get_lines(), align=False, fontsize=labelfont,
                      xvals = [.09,.04,.04],
                      yoffsets=[-.08,0.09, -.08],
                      bbox={'alpha': 0},
                            )

fig.tight_layout()

plt.savefig(imagedir+'prop2-uni.pdf')

#%% Minority representation in size-1 juries (fig:counter)

fig,(ax,ax2) = plt.subplots(1,2, figsize=(figsize,figsize/2+.1))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(-0.01,1.01)
ax.set_ylim(-0.05,1.9)
ax.set_xlabel('Conviction probability $c$')
ax.set_yticks(np.arange(0,2,0.5))
xx = np.linspace(0,1,1000)
SAR = np.ones(len(xx))*.75
SAR[374:625]=1.75
RAN =  np.ones(len(xx))

STR = 6*xx*(1-xx)

ax.set_ylabel('Density', fontsize=11)
ax.set_xlabel('Conviction probability $c$', fontsize=xylabelfont)

ax.plot(xx, RAN, ':', color=RANcolor, linewidth=2, label=ran_label)
ax.plot(xx, SAR, '', color=SARcolor, label=rep_label)
ax.plot(xx, STR, '--', color=STRcolor, label=str_label)

labellines.labelLines(ax.get_lines(), align=False, fontsize=labelfont,
                      xvals = [.91,.3,.2],
                      yoffsets=[.1,-0.15,.4],
                      bbox={'alpha': 0},
                            )

# Minority representation in size-1 juries (fig:counter_b)

fname = gzip.open(outputdir+'uni-1-1-1-MANYr.gz','rb')
resultall = pickle.load(fname)
fname.close()

xvalue = 7
Rvector = np.array([])
avRAN = np.array([])
avSAR = np.array([])
avSTR = np.array([])

for model in resultall:
    baseargs = model['baseargs']
    Rvector = np.append(Rvector,1-baseargs['R'])
    avRAN = np.append(avRAN,np.average(model['juriestRAN']==0))
    avSAR = np.append(avSAR,np.average(model['juriestSAR']==0))
    avSTR = np.append(avSTR,np.average(model['juriestSTR']==0))


#fig,ax = plt.subplots(figsize=(2.5,2.5))
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_xlim(0,.42)
ax2.set_ylabel('Frac. group-a in juries', fontsize = xylabelfont)
ax2.set_xlabel('Fraction group-a in pool', fontsize = xylabelfont)
ax2.set_yticks(np.arange(0,0.41,0.1))
ax2.set_xticks(np.arange(0,0.41,0.1))

ax2.plot(Rvector, avRAN, ':', color=RANcolor, linewidth=2, label=ran_label)
ax2.plot(Rvector, avSAR, '', color=SARcolor, label=rep_label)
ax2.plot(Rvector, avSTR, '--', color=STRcolor, label=str_label)

fig.tight_layout()
labellines.labelLines(ax2.get_lines(), align=False, fontsize=labelfont,
                      xvals = [.1,.38,.2],
                      yoffsets=[.05,-0.052, -.05],
                      bbox={'alpha': 0},
                            )

ax.set_title('(a) Selected juror $c$ distribution',fontsize=xylabelfont)
ax2.set_title('(b) Minority representation',fontsize=xylabelfont)

fig.tight_layout()
plt.savefig(imagedir+'counterall.pdf')
#plt.savefig(imagedir+'counter_b.pdf')

#%% Representation of minority jurors (tab:betas-grouprep - Table 1)

fname = gzip.open(outputdir+'beta-1-5-12j-50pcT1.pickle.gz','rb')
resultbeta_50 = pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-1-5-12j-75pcT1.pickle.gz','rb')
resultbeta_75 =pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-1-5-12j-90pcT1.pickle.gz','rb')
resultbeta_90 = pickle.load(fname)
fname.close()

#compute statistics of  number type-0 of jurors
#CHANGE T
avSAR_50_b15 = np.average(np.average(1-resultbeta_50['juriestSAR'],axis=1))
avSTR_50_b15 = np.average(np.average(1-resultbeta_50['juriestSTR'],axis=1))
avRAN_50_b15 = np.average(np.average(1-resultbeta_50['juriestRAN'],axis=1))
sdSAR_50_b15 = np.std(np.average(1-resultbeta_50['juriestSAR'],axis=1))
sdSTR_50_b15 = np.std(np.average(1-resultbeta_50['juriestSTR'],axis=1))
sdRAN_50_b15 = np.std(np.average(1-resultbeta_50['juriestRAN'],axis=1))
ng1SAR_50_b15 = np.average(np.sum(1-resultbeta_50['juriestSAR'],axis=1)>=1)
ng1STR_50_b15 = np.average(np.sum(1-resultbeta_50['juriestSTR'],axis=1)>=1)
ng1RAN_50_b15 = np.average(np.sum(1-resultbeta_50['juriestRAN'],axis=1)>=1)

avSAR_75_b15 =np.average(np.average(1-resultbeta_75['juriestSAR'],axis=1))
avSTR_75_b15 =np.average(np.average(1-resultbeta_75['juriestSTR'],axis=1))
avRAN_75_b15 =np.average(np.average(1-resultbeta_75['juriestRAN'],axis=1))
sdSAR_75_b15 =np.std(np.average(1-resultbeta_75['juriestSAR'],axis=1))
sdSTR_75_b15 =np.std(np.average(1-resultbeta_75['juriestSTR'],axis=1))
sdRAN_75_b15 =np.std(np.average(1-resultbeta_75['juriestRAN'],axis=1))
ng1SAR_75_b15 =np.average(np.sum(1-resultbeta_75['juriestSAR'],axis=1)>=1)
ng1STR_75_b15 =np.average(np.sum(1-resultbeta_75['juriestSTR'],axis=1)>=1)
ng1RAN_75_b15 =np.average(np.sum(1-resultbeta_75['juriestRAN'],axis=1)>=1)

avSAR_90_b15 = np.average(np.average(1-resultbeta_90['juriestSAR'],axis=1))
avSTR_90_b15 = np.average(np.average(1-resultbeta_90['juriestSTR'],axis=1))
avRAN_90_b15 = np.average(np.average(1-resultbeta_90['juriestRAN'],axis=1))
sdSAR_90_b15 = np.std(np.average(1-resultbeta_90['juriestSAR'],axis=1))
sdSTR_90_b15 = np.std(np.average(1-resultbeta_90['juriestSTR'],axis=1))
sdRAN_90_b15 = np.std(np.average(1-resultbeta_90['juriestRAN'],axis=1))
ng1SAR_90_b15 = np.average(np.sum(1-resultbeta_90['juriestSAR'],axis=1)>=1)
ng1STR_90_b15 = np.average(np.sum(1-resultbeta_90['juriestSTR'],axis=1)>=1)
ng1RAN_90_b15 = np.average(np.sum(1-resultbeta_90['juriestRAN'],axis=1)>=1)

fname = gzip.open(outputdir+'beta-2-4-12j-50pcT1.pickle.gz','rb')
resultbeta_50 = pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-2-4-12j-75pcT1.pickle.gz','rb')
resultbeta_75 =pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-2-4-12j-90pcT1.pickle.gz','rb')
resultbeta_90 = pickle.load(fname)
fname.close()

#compute statistics of  number type-0 of jurors
avSAR_50_b24 = np.average(np.average(1-resultbeta_50['juriestSAR'],axis=1))
avSTR_50_b24 = np.average(np.average(1-resultbeta_50['juriestSTR'],axis=1))
avRAN_50_b24 = np.average(np.average(1-resultbeta_50['juriestRAN'],axis=1))
sdSAR_50_b24 = np.std(np.average(1-resultbeta_50['juriestSAR'],axis=1))
sdSTR_50_b24 = np.std(np.average(1-resultbeta_50['juriestSTR'],axis=1))
sdRAN_50_b24 = np.std(np.average(1-resultbeta_50['juriestRAN'],axis=1))
ng1SAR_50_b24 = np.average(np.sum(1-resultbeta_50['juriestSAR'],axis=1)>=1)
ng1STR_50_b24 = np.average(np.sum(1-resultbeta_50['juriestSTR'],axis=1)>=1)
ng1RAN_50_b24 = np.average(np.sum(1-resultbeta_50['juriestRAN'],axis=1)>=1)

avSAR_75_b24 =np.average(np.average(1-resultbeta_75['juriestSAR'],axis=1))
avSTR_75_b24 =np.average(np.average(1-resultbeta_75['juriestSTR'],axis=1))
avRAN_75_b24 =np.average(np.average(1-resultbeta_75['juriestRAN'],axis=1))
sdSAR_75_b24 =np.std(np.average(1-resultbeta_75['juriestSAR'],axis=1))
sdSTR_75_b24 =np.std(np.average(1-resultbeta_75['juriestSTR'],axis=1))
sdRAN_75_b24 =np.std(np.average(1-resultbeta_75['juriestRAN'],axis=1))
ng1SAR_75_b24 =np.average(np.sum(1-resultbeta_75['juriestSAR'],axis=1)>=1)
ng1STR_75_b24 =np.average(np.sum(1-resultbeta_75['juriestSTR'],axis=1)>=1)
ng1RAN_75_b24 =np.average(np.sum(1-resultbeta_75['juriestRAN'],axis=1)>=1)

avSAR_90_b24 = np.average(np.average(1-resultbeta_90['juriestSAR'],axis=1))
avSTR_90_b24 = np.average(np.average(1-resultbeta_90['juriestSTR'],axis=1))
avRAN_90_b24 = np.average(np.average(1-resultbeta_90['juriestRAN'],axis=1))
sdSAR_90_b24 = np.std(np.average(1-resultbeta_90['juriestSAR'],axis=1))
sdSTR_90_b24 = np.std(np.average(1-resultbeta_90['juriestSTR'],axis=1))
sdRAN_90_b24 = np.std(np.average(1-resultbeta_90['juriestRAN'],axis=1))
ng1SAR_90_b24 = np.average(np.sum(1-resultbeta_90['juriestSAR'],axis=1)>=1)
ng1STR_90_b24 = np.average(np.sum(1-resultbeta_90['juriestSTR'],axis=1)>=1)
ng1RAN_90_b24 = np.average(np.sum(1-resultbeta_90['juriestRAN'],axis=1)>=1)

fname = gzip.open(outputdir+'beta-3-4-12j-50pcT1.pickle.gz','rb')
resultbeta_50 = pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-3-4-12j-75pcT1.pickle.gz','rb')
resultbeta_75 =pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-3-4-12j-90pcT1.pickle.gz','rb')
resultbeta_90 = pickle.load(fname)
fname.close()

#compute statistics of  number type-0 of jurors
avSAR_50_b34 = np.average(np.average(1-resultbeta_50['juriestSAR'],axis=1))
avSTR_50_b34 = np.average(np.average(1-resultbeta_50['juriestSTR'],axis=1))
avRAN_50_b34 = np.average(np.average(1-resultbeta_50['juriestRAN'],axis=1))
sdSAR_50_b34 = np.std(np.average(1-resultbeta_50['juriestSAR'],axis=1))
sdSTR_50_b34 = np.std(np.average(1-resultbeta_50['juriestSTR'],axis=1))
sdRAN_50_b34 = np.std(np.average(1-resultbeta_50['juriestRAN'],axis=1))
ng1SAR_50_b34 = np.average(np.sum(1-resultbeta_50['juriestSAR'],axis=1)>=1)
ng1STR_50_b34 = np.average(np.sum(1-resultbeta_50['juriestSTR'],axis=1)>=1)
ng1RAN_50_b34 = np.average(np.sum(1-resultbeta_50['juriestRAN'],axis=1)>=1)

avSAR_75_b34 =np.average(np.average(1-resultbeta_75['juriestSAR'],axis=1))
avSTR_75_b34 =np.average(np.average(1-resultbeta_75['juriestSTR'],axis=1))
avRAN_75_b34 =np.average(np.average(1-resultbeta_75['juriestRAN'],axis=1))
sdSAR_75_b34 =np.std(np.average(1-resultbeta_75['juriestSAR'],axis=1))
sdSTR_75_b34 =np.std(np.average(1-resultbeta_75['juriestSTR'],axis=1))
sdRAN_75_b34 =np.std(np.average(1-resultbeta_75['juriestRAN'],axis=1))
ng1SAR_75_b34 =np.average(np.sum(1-resultbeta_75['juriestSAR'],axis=1)>=1)
ng1STR_75_b34 =np.average(np.sum(1-resultbeta_75['juriestSTR'],axis=1)>=1)
ng1RAN_75_b34 =np.average(np.sum(1-resultbeta_75['juriestRAN'],axis=1)>=1)

avSAR_90_b34 = np.average(np.average(1-resultbeta_90['juriestSAR'],axis=1))
avSTR_90_b34 = np.average(np.average(1-resultbeta_90['juriestSTR'],axis=1))
avRAN_90_b34 = np.average(np.average(1-resultbeta_90['juriestRAN'],axis=1))
sdSAR_90_b34 = np.std(np.average(1-resultbeta_90['juriestSAR'],axis=1))
sdSTR_90_b34 = np.std(np.average(1-resultbeta_90['juriestSTR'],axis=1))
sdRAN_90_b34 = np.std(np.average(1-resultbeta_90['juriestRAN'],axis=1))
ng1SAR_90_b34 = np.average(np.sum(1-resultbeta_90['juriestSAR'],axis=1)>=1)
ng1STR_90_b34 = np.average(np.sum(1-resultbeta_90['juriestSTR'],axis=1)>=1)
ng1RAN_90_b34 = np.average(np.sum(1-resultbeta_90['juriestRAN'],axis=1)>=1)

texstring = '\n'
texstring += 'Group-a represents 50\% of the jury pool \n\n'
texstring += 'Average fraction of minorities \t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' %           (avSAR_50_b15,avSTR_50_b15,avSAR_50_b24,avSTR_50_b24,avSAR_50_b34,avSTR_50_b34,avRAN_50_b34)
texstring += '\nStandard deviation  \t\t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' %     (sdSAR_50_b15,sdSTR_50_b15,sdSAR_50_b24,sdSTR_50_b24,sdSAR_50_b34,sdSTR_50_b34,sdRAN_50_b34)
texstring += '\nFraction of juries with at least 1 \t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' % (ng1SAR_50_b15,ng1STR_50_b15,ng1SAR_50_b24,ng1STR_50_b24,ng1SAR_50_b34,ng1STR_50_b34,ng1RAN_50_b34)

texstring += '\n\n'
texstring += 'Group-a represents 25\% of the jury pool \n\n'
texstring += 'Average fraction of minorities \t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' %           (avSAR_75_b15,avSTR_75_b15,avSAR_75_b24,avSTR_75_b24,avSAR_75_b34,avSTR_75_b34,avRAN_75_b34)
texstring += '\nStandard deviation  \t\t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' %     (sdSAR_75_b15,sdSTR_75_b15,sdSAR_75_b24,sdSTR_75_b24,sdSAR_75_b34,sdSTR_75_b34,sdRAN_75_b34)
texstring += '\nFraction of juries with at least 1 \t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' % (ng1SAR_75_b15,ng1STR_75_b15,ng1SAR_75_b24,ng1STR_75_b24,ng1SAR_75_b34,ng1STR_75_b34,ng1RAN_75_b34)

texstring += '\n\n'
texstring += 'Group-a represents 10\% of the jury pool \n\n'
texstring += 'Average fraction of minorities \t & \t %3.2f  \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' %           (avSAR_90_b15,avSTR_90_b15,avSAR_90_b24,avSTR_90_b24,avSAR_90_b34,avSTR_90_b34,avRAN_90_b34)
texstring += '\nStandard deviation  \t\t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' %     (sdSAR_90_b15,sdSTR_90_b15,sdSAR_90_b24,sdSTR_90_b24,sdSAR_90_b34,sdSTR_90_b34,sdRAN_90_b34)
texstring += '\nFraction of juries with at least 1 \t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' % (ng1SAR_90_b15,ng1STR_90_b15,ng1SAR_90_b24,ng1STR_90_b24,ng1SAR_90_b34,ng1STR_90_b34,ng1RAN_90_b34)
print(texstring)

with open(imagedir + 'tables.tex', 'w') as outputfile:
    outputfile.write(texstring)
outputfile.close()

#%% Barplot version of table above for revision

hatch = ['..','\\\\\\','']
hatch2 = ['xx','///','++']
hatchrep = ['\\\\\\']*3
hatchstr = ['..']*3
hatch2rep = ['///']
hatch2str = ['xx']*3

def new_func(ax1, values, valgr1):

    # generate percent value and round to integer
    for mod,val in values.items():
        values[mod]= [np.round(item*100) for item in val]
    for mod,val in valgr1.items():
        valgr1[mod]= [np.round(item*100) for item in val]
    width = 0.2  # the width of the bars
    multiplier = 0.5
    key = 0

    offset = width * multiplier
    rects = ax1.bar(np.arange(4) + offset, values['REP'], width, label=['REP']*4, hatch=hatchrep+[''], color=barcolor2[key%2], edgecolor = edgecolor)
    ax1.bar_label(rects, padding=3, fontsize=labelfont)
    multiplier += 1

    key = 1
    offset = width * multiplier
    rects = ax1.bar(np.array([0,1,2,3+width/4])+offset, values['STR'], width, label=['STR']*4, hatch=hatchstr+['---'], color=barcolor2[key%2], edgecolor = edgecolor)
    ax1.bar_label(rects, padding=3, fontsize=labelfont)
    multiplier += 1

    key = 0
    offset = width * multiplier + width/4
    rects2 = ax1.bar(np.arange(3) + offset, valgr1['REP'], width, label=['REP']*3, hatch=hatch2rep, color=barcolor2[key%2], edgecolor = edgecolor)
    ax1.bar_label(rects2, padding=3, fontsize=labelfont)
    multiplier += 1

    key = 1
    offset = width * multiplier + width/4
    rects2 = ax1.bar(np.arange(3) + offset, valgr1['STR'], width, label=['STR']*3, hatch=hatch2str, color=barcolor2[key%2], edgecolor = edgecolor)
    ax1.bar_label(rects2, padding=3, fontsize=labelfont)

    tixlocs = [.4, 1.4, 2.4, 3.2]
    ax1.set_xticks(tixlocs,['Extreme', 'Moderate', 'Mild', '(All)'], fontsize = labelfont)
    return ax1

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(figsize,figsize))

for ax in [ax1,ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.tick_params(length=0)
    ax.set_ylim(0,115)

values = {'REP': [avSAR_75_b15,avSAR_75_b24,avSAR_75_b34,avRAN_75_b15 ], # [.1, .18,.23], 
          'STR': [avSTR_75_b15,avSTR_75_b24, avSTR_75_b34, ng1RAN_75_b15], #[.08, .16, .23],
          'RAN': [avRAN_75_b15], #[.25],
}
# % juries with minorities
valgr1 = {'REP': [ng1SAR_75_b15,ng1SAR_75_b24,ng1SAR_75_b34,], #[.57, .88, .96],
          'STR': [ng1STR_75_b15,ng1STR_75_b24,ng1STR_75_b34], #[.45, .84, .95],
          'RAN': [ng1RAN_75_b15], #[.97],
          }

ax1=new_func(ax1, values, valgr1)

values = {'REP': [avSAR_90_b15,avSAR_90_b24,avSAR_90_b34, avRAN_90_b15], # [.1, .18,.23], 
          'STR': [avSTR_90_b15,avSTR_90_b24, avSTR_90_b34, ng1RAN_90_b15], #[.08, .16, .23],
          'RAN': [avRAN_90_b15], #[.25],
}

valgr1 = {'REP': [ng1SAR_90_b15,ng1SAR_90_b24,ng1SAR_90_b34], # [.17, .47, .67],
          'STR': [ng1STR_90_b15,ng1STR_90_b24,ng1STR_90_b34], #[.02, .38, .64],
          'RAN': [ng1RAN_90_b15], #[.72],
          }

ax2=new_func(ax2, values, valgr1)


# generate percent value and round to integer
for mod,val in values.items():
    values[mod]= [np.round(item*100) for item in val]
for mod,val in valgr1.items():
    valgr1[mod]= [np.round(item*100) for item in val]

#Add some text for labels, title and custom x-axis tick labels, etc.
handles, labels = ax2.get_legend_handles_labels()
firstlegend = ax2.legend([handles[0],handles[4],handles[3]], [rep_label,str_label,ran_label], 
                         loc='upper left',  bbox_to_anchor=(0,1.35), title=r'\% minorities in juries', 
                         title_fontsize=labelfont, 
                         ncol=2, fontsize=labelfont)
secondlegend = ax2.legend([handles[8],handles[12],handles[7]], [rep_label,str_label,ran_label], 
                          loc='upper left', bbox_to_anchor=(0.4,1.35), title=r'\% juries with minorities', 
                          ncol=2, title_fontsize=labelfont, fontsize = labelfont)
for t in secondlegend.get_texts():
    t.set_ha('left')
ax2.add_artist(firstlegend)

ax1.set_title('(a) Group-a size 25\%', fontsize=xylabelfont)
ax2.text(.2*5+.1,85,'(b) Group-a size 10\%', fontsize=xylabelfont)

fig.tight_layout()
plt.savefig(imagedir+'minority-representation.pdf')

#%% Number of challenges, (fig:nchallenges)

if __name__ == '__main__':
    fname = gzip.open(outputdir+'manyChallenges.pickle.gz','rb')
    allchg = pickle.load(fname)
    fname.close()

    nLess_c_SAR = np.array([])
    nLess_c_STR = np.array([])
    minSAR = np.array([])
    minSTR = np.array([])
    challenges = np.array([],dtype=int)

    xx = np.arange(0, 1.01, 0.01)
    pct = np.array([])
    for c in xx:
        pct = np.append(pct, np.count_nonzero(allchg[0]['juriesxRAN']<c) /allchg[0]['juriesxRAN'].size)

    allextremes = list()
    for juriesoutp in allchg:
        challenges = np.append(challenges,juriesoutp['baseargs']['D'])
        xSAR = juriesoutp['juriesxSAR']
        xSTR = juriesoutp['juriesxSTR']
        tSAR = juriesoutp['juriestSAR']
        tSTR = juriesoutp['juriestSTR']
        njuries = xSAR.shape[0]

        # exclusion of extremes
        c = 0.27
        nLess_c_SAR = np.append(nLess_c_SAR, np.sum(np.count_nonzero(xSAR<c, axis=1) >=1 ) /njuries)
        nLess_c_STR = np.append(nLess_c_STR, np.sum(np.count_nonzero(xSTR<c, axis=1) >=1 ) /njuries)

        # minority representation
        minSAR = np.append(minSAR,1-np.average(tSAR))
        minSTR = np.append(minSTR,1-np.average(tSTR))

        # fraction of juries less than many thresholds, STR-SAR difference
        cx = np.arange(0.01,0.45,.01)

        extremes = np.zeros(cx.size)
        for idx,c in enumerate(cx):
            extremes[idx] = (np.sum(np.count_nonzero(xSTR<c, axis=1) >=1 )/njuries -
                             np.sum(np.count_nonzero(xSAR<c, axis=1) >=1 )/njuries )
        allextremes.append(extremes)
    allextremes = np.array(allextremes)

#%%
if __name__ == '__main__':
    # now draw the figures
    fig,(ax,ax2) = plt.subplots(1,2,figsize=(figsize,figsize/2+.2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(0,len(challenges)+1,2))
    ax.set_xlabel('Number of challenges', fontsize=xylabelfont)
    ax.set_ylabel('Fraction of juries', fontsize=xylabelfont)
    ax.plot(challenges, nLess_c_SAR, marker='s', color=SARcolor, label=rep_label, linewidth=.6, markersize=3)
    ax.plot(challenges, nLess_c_STR, '--', marker='o',fillstyle='none', color=STRcolor, label=str_label, linewidth=.6, markersize=3)
    labellines.labelLines(ax.get_lines(), align=False, yoffsets=[.095,.033], bbox={'alpha': 0}, fontsize=labelfont, )# align=False)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xticks(np.arange(0,len(challenges)+1,2))
    ax2.set_xlabel('Number of challenges', fontsize = xylabelfont)
    ax2.set_ylabel('Fraction of minorities', fontsize = xylabelfont)
    ax2.plot(challenges, minSAR, marker='s', color=SARcolor, label=rep_label, linewidth=.6, markersize=3)
    ax2.plot(challenges, minSTR, '--', marker='o',fillstyle='none', color=STRcolor, label=str_label, linewidth=.4, markersize=3)
    labellines.labelLines(ax2.get_lines(), align=False, xvals=[7,8],yoffsets=[.015,-.012], bbox={'alpha': 0}, fontsize=labelfont, )# align=False)
    
    ax.set_title('(a) Juries with at least 1 extreme', fontsize=xylabelfont)
    ax2.set_title('(b) Minority representation', fontsize=xylabelfont)
    ax2.set_ylim(.08,.19)
    ax2.set_yticks(np.arange(.1,.2,.02))
    fig.tight_layout()
    plt.savefig(imagedir+'nchallenges-extr-minority.pdf')

    #%% Prob. of selecting x jurors abobe median (fig:median)
fname = gzip.open(outputdir+'beta-1-5-12j-50pcT1.pickle.gz','rb')
resultbeta_50 = pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-1-5-12j-75pcT1.pickle.gz','rb')
resultbeta_75 = pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-1-5-12j-90pcT1.pickle.gz','rb')
resultbeta_90 = pickle.load(fname)
fname.close()

#get median

#resultbeta_50['juriestRAN'],axis=1)
median50 = np.median(resultbeta_50['juriesxRAN'])
median75 = np.median(resultbeta_75['juriesxRAN'])
median90 = np.median(resultbeta_90['juriesxRAN'])

# compute prob that n jurors are above mediand
nSARgmed_50 = np.sum(resultbeta_50['juriesxSAR']<median50,axis=1)
nSARgmed_75 = np.sum(resultbeta_75['juriesxSAR']<median75,axis=1)
nSARgmed_90 = np.sum(resultbeta_90['juriesxSAR']<median90,axis=1)
nSTRgmed_50 = np.sum(resultbeta_50['juriesxSTR']<median50,axis=1)
nSTRgmed_75 = np.sum(resultbeta_75['juriesxSTR']<median75,axis=1)
nSTRgmed_90 = np.sum(resultbeta_90['juriesxSTR']<median90,axis=1)
nRANgmed_90 = np.sum(resultbeta_90['juriesxRAN']<median90,axis=1)
nRANgmed_50 = np.sum(resultbeta_50['juriesxRAN']<median50,axis=1)
nRANgmed_75 = np.sum(resultbeta_75['juriesxRAN']<median75,axis=1)

njurors = resultbeta_50['juriesxRAN'].shape[1]
prgmed50SAR = np.zeros(njurors)
prgmed75SAR = np.zeros(njurors)
prgmed90SAR = np.zeros(njurors)
prgmed50STR = np.zeros(njurors)
prgmed75STR = np.zeros(njurors)
prgmed90STR = np.zeros(njurors)
prgmed50RAN = np.zeros(njurors)
prgmed75RAN = np.zeros(njurors)
prgmed90RAN = np.zeros(njurors)
for n in range(6,13):
    prgmed50SAR[n-1] = np.sum(nSARgmed_50>=n)/len(nSARgmed_50)
    prgmed75SAR[n-1] = np.sum(nSARgmed_75>=n)/len(nSARgmed_75)
    prgmed90SAR[n-1] = np.sum(nSARgmed_90>=n)/len(nSARgmed_90)
    prgmed50STR[n-1] = np.sum(nSTRgmed_50>=n)/len(nSTRgmed_50)
    prgmed75STR[n-1] = np.sum(nSTRgmed_75>=n)/len(nSTRgmed_75)
    prgmed90STR[n-1] = np.sum(nSTRgmed_90>=n)/len(nSTRgmed_90)
    prgmed50RAN[n-1] = np.sum(nRANgmed_50>=n)/len(nRANgmed_50)
    prgmed75RAN[n-1] = np.sum(nRANgmed_75>=n)/len(nRANgmed_75)
    prgmed90RAN[n-1] = np.sum(nRANgmed_90>=n)/len(nRANgmed_90)

print('beta 1-5')
print('medians [50,75,90]:',median50,median75,median90)
print('SAR, 50%T1', prgmed50SAR[6:12])
print('STR, 50%T1', prgmed50STR[6:12])
print('SAR, 75%T1', prgmed75SAR[6:12])
print('STR, 75%T1', prgmed75STR[6:12])
print('SAR, 90%T1', prgmed90SAR[6:12])
print('STR, 90%T1', prgmed90STR[6:12])

x = np.arange(7,13)
fig,ax = plt.subplots(figsize=(4,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(6.8,12.05)
ax.set_ylim(-0.02,0.09)
ax.set_xlabel('Min. number of jurors below the median',fontsize=xylabelfont)
ax.set_ylabel('Fraction of juries (diff. with \\textit{RAN})',fontsize=xylabelfont)
shift=.2

ax.plot(x,prgmed50STR[6:13]-prgmed50RAN[6:13],'--',marker='*',color=STRcolor,label=str_label,linewidth=.8)
ax.plot(x,prgmed90SAR[6:13]-prgmed90RAN[6:13],marker='s',color=RANcolor,label=r10,linewidth=.8, markersize=4)
ax.plot(x,prgmed75SAR[6:13]-prgmed75RAN[6:13],':',marker='o',color=SARcolor,label=r25,linewidth=.8, markersize=4)
ax.plot(x,prgmed50SAR[6:13]-prgmed50RAN[6:13],'-.',marker='^',color=alt2color,label=r50,linewidth=.8, markersize=4)
fig.tight_layout()
#ax.legend()
labellines.labelLines(plt.gca().get_lines(), align=False, fontsize=labelfont,
                      xvals = [10.5,7.3,8.35,8.5],
                      yoffsets=[0.012,+0.01,-0.012,-0.016],
                      bbox={'alpha': 0},
                            )
plt.savefig(imagedir+'median.pdf')


#%% Representation in balanced groups  (table tab:betas-grouprep-balanced)

#(Table 2 (b): Slightly Asymmetric distributions
fname = gzip.open(outputdir+'beta-1-5-5-2-12j-50pcT1.pickle.gz','rb')
resultbeta_50_1_5_tilt = pickle.load(fname)
fname.close()

#compute statistics of  number type-0 of jurors
avSAR_50_1_5_tilt = np.average(np.average(1-resultbeta_50_1_5_tilt['juriestSAR'],axis=1))
avSTR_50_1_5_tilt = np.average(np.average(1-resultbeta_50_1_5_tilt['juriestSTR'],axis=1))
avRAN_50_1_5_tilt = np.average(np.average(1-resultbeta_50_1_5_tilt['juriestRAN'],axis=1))
sdSAR_50_1_5_tilt = np.std(np.average(1-resultbeta_50_1_5_tilt['juriestSAR'],axis=1))
sdSTR_50_1_5_tilt = np.std(np.average(1-resultbeta_50_1_5_tilt['juriestSTR'],axis=1))
sdRAN_50_1_5_tilt = np.std(np.average(1-resultbeta_50_1_5_tilt['juriestRAN'],axis=1))
ng1SAR_50_1_5_tilt = np.average(np.sum(1-resultbeta_50_1_5_tilt['juriestSAR'],axis=1)>=1)
ng1STR_50_1_5_tilt = np.average(np.sum(1-resultbeta_50_1_5_tilt['juriestSTR'],axis=1)>=1)
ng1RAN_50_1_5_tilt = np.average(np.sum(1-resultbeta_50_1_5_tilt['juriestRAN'],axis=1)>=1)

fname = gzip.open(outputdir+'beta-2-4-4-3-12j-50pcT1.pickle.gz','rb')
resultbeta_50_2_4_tilt = pickle.load(fname)
fname.close()

#compute statistics of  number type-0 of jurors
avSAR_50_2_4_tilt = np.average(np.average(1-resultbeta_50_2_4_tilt['juriestSAR'],axis=1))
avSTR_50_2_4_tilt = np.average(np.average(1-resultbeta_50_2_4_tilt['juriestSTR'],axis=1))
avRAN_50_2_4_tilt = np.average(np.average(1-resultbeta_50_2_4_tilt['juriestRAN'],axis=1))
sdSAR_50_2_4_tilt = np.std(np.average(1-resultbeta_50_2_4_tilt['juriestSAR'],axis=1))
sdSTR_50_2_4_tilt = np.std(np.average(1-resultbeta_50_2_4_tilt['juriestSTR'],axis=1))
sdRAN_50_2_4_tilt = np.std(np.average(1-resultbeta_50_2_4_tilt['juriestRAN'],axis=1))
ng1SAR_50_2_4_tilt = np.average(np.sum(1-resultbeta_50_2_4_tilt['juriestSAR'],axis=1)>=1)
ng1STR_50_2_4_tilt = np.average(np.sum(1-resultbeta_50_2_4_tilt['juriestSTR'],axis=1)>=1)
ng1RAN_50_2_4_tilt = np.average(np.sum(1-resultbeta_50_2_4_tilt['juriestRAN'],axis=1)>=1)

fname = gzip.open(outputdir+'beta-2-4-4-3-12j-50pcT1.pickle.gz','rb')
resultbeta_50_3_4_tilt = pickle.load(fname)
fname.close()

#compute statistics of  number type-0 of jurors
avSAR_50_3_4_tilt = np.average(np.average(1-resultbeta_50_3_4_tilt['juriestSAR'],axis=1))
avSTR_50_3_4_tilt = np.average(np.average(1-resultbeta_50_3_4_tilt['juriestSTR'],axis=1))
avRAN_50_3_4_tilt = np.average(np.average(1-resultbeta_50_3_4_tilt['juriestRAN'],axis=1))
sdSAR_50_3_4_tilt = np.std(np.average(1-resultbeta_50_3_4_tilt['juriestSAR'],axis=1))
sdSTR_50_3_4_tilt = np.std(np.average(1-resultbeta_50_3_4_tilt['juriestSTR'],axis=1))
sdRAN_50_3_4_tilt = np.std(np.average(1-resultbeta_50_3_4_tilt['juriestRAN'],axis=1))
ng1SAR_50_3_4_tilt = np.average(np.sum(1-resultbeta_50_3_4_tilt['juriestSAR'],axis=1)>=1)
ng1STR_50_3_4_tilt = np.average(np.sum(1-resultbeta_50_3_4_tilt['juriestSTR'],axis=1)>=1)
ng1RAN_50_3_4_tilt = np.average(np.sum(1-resultbeta_50_3_4_tilt['juriestRAN'],axis=1)>=1)

# Table 2 (c) : Slightly Asymetric groups
fname = gzip.open(outputdir+'beta-1-5-12j-55pcT1.pickle.gz','rb')
resultbeta_50_1_5_unbal = pickle.load(fname)
fname.close()

#compute statistics of  number type-0 of jurors
avSAR_50_1_5_unbal = np.average(np.average(1-resultbeta_50_1_5_unbal['juriestSAR'],axis=1))
avSTR_50_1_5_unbal = np.average(np.average(1-resultbeta_50_1_5_unbal['juriestSTR'],axis=1))
avRAN_50_1_5_unbal = np.average(np.average(1-resultbeta_50_1_5_unbal['juriestRAN'],axis=1))
sdSAR_50_1_5_unbal = np.std(np.average(1-resultbeta_50_1_5_unbal['juriestSAR'],axis=1))
sdSTR_50_1_5_unbal = np.std(np.average(1-resultbeta_50_1_5_unbal['juriestSTR'],axis=1))
sdRAN_50_1_5_unbal = np.std(np.average(1-resultbeta_50_1_5_unbal['juriestRAN'],axis=1))
ng1SAR_50_1_5_unbal = np.average(np.sum(1-resultbeta_50_1_5_unbal['juriestSAR'],axis=1)>=1)
ng1STR_50_1_5_unbal = np.average(np.sum(1-resultbeta_50_1_5_unbal['juriestSTR'],axis=1)>=1)
ng1RAN_50_1_5_unbal = np.average(np.sum(1-resultbeta_50_1_5_unbal['juriestRAN'],axis=1)>=1)

fname = gzip.open(outputdir+'beta-2-4-12j-55pcT1.pickle.gz','rb')
resultbeta_50_2_4_unbal = pickle.load(fname)
fname.close()

#compute statistics of  number type-0 of jurors
avSAR_50_2_4_unbal = np.average(np.average(1-resultbeta_50_2_4_unbal['juriestSAR'],axis=1))
avSTR_50_2_4_unbal = np.average(np.average(1-resultbeta_50_2_4_unbal['juriestSTR'],axis=1))
avRAN_50_2_4_unbal = np.average(np.average(1-resultbeta_50_2_4_unbal['juriestRAN'],axis=1))
sdSAR_50_2_4_unbal = np.std(np.average(1-resultbeta_50_2_4_unbal['juriestSAR'],axis=1))
sdSTR_50_2_4_unbal = np.std(np.average(1-resultbeta_50_2_4_unbal['juriestSTR'],axis=1))
sdRAN_50_2_4_unbal = np.std(np.average(1-resultbeta_50_2_4_unbal['juriestRAN'],axis=1))
ng1SAR_50_2_4_unbal = np.average(np.sum(1-resultbeta_50_2_4_unbal['juriestSAR'],axis=1)>=1)
ng1STR_50_2_4_unbal = np.average(np.sum(1-resultbeta_50_2_4_unbal['juriestSTR'],axis=1)>=1)
ng1RAN_50_2_4_unbal = np.average(np.sum(1-resultbeta_50_2_4_unbal['juriestRAN'],axis=1)>=1)

fname = gzip.open(outputdir+'beta-3-4-12j-55pcT1.pickle.gz','rb')
resultbeta_50_3_4_unbal = pickle.load(fname)
fname.close()

#compute statistics of number type-0 of jurors
avSAR_50_3_4_unbal = np.average(np.average(1-resultbeta_50_3_4_unbal['juriestSAR'],axis=1))
avSTR_50_3_4_unbal = np.average(np.average(1-resultbeta_50_3_4_unbal['juriestSTR'],axis=1))
avRAN_50_3_4_unbal = np.average(np.average(1-resultbeta_50_3_4_unbal['juriestRAN'],axis=1))
sdSAR_50_3_4_unbal = np.std(np.average(1-resultbeta_50_3_4_unbal['juriestSAR'],axis=1))
sdSTR_50_3_4_unbal = np.std(np.average(1-resultbeta_50_3_4_unbal['juriestSTR'],axis=1))
sdRAN_50_3_4_unbal = np.std(np.average(1-resultbeta_50_3_4_unbal['juriestRAN'],axis=1))
ng1SAR_50_3_4_unbal = np.average(np.sum(1-resultbeta_50_3_4_unbal['juriestSAR'],axis=1)>=1)
ng1STR_50_3_4_unbal = np.average(np.sum(1-resultbeta_50_3_4_unbal['juriestSTR'],axis=1)>=1)
ng1RAN_50_3_4_unbal = np.average(np.sum(1-resultbeta_50_3_4_unbal['juriestRAN'],axis=1)>=1)

# Print into Tables 
texstring = "\nRepresentation in balanced groups  (table tab:betas-grouprep-balanced) \n\n"
texstring += table_pre
texstring += '     \\multicolumn{8}{c}{$r=0.45$}\n'
texstring += 'Average fraction of minorities \t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' %           (avSAR_50_1_5_unbal,avSTR_50_1_5_unbal,avSAR_50_2_4_unbal,avSTR_50_2_4_unbal,avSAR_50_3_4_unbal,avSTR_50_3_4_unbal,avRAN_50_3_4_unbal)
texstring += '\nStandard deviation  \t\t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' %     (sdSAR_50_1_5_unbal,sdSTR_50_1_5_unbal,sdSAR_50_2_4_unbal,sdSTR_50_2_4_unbal,sdSAR_50_3_4_unbal,sdSTR_50_3_4_unbal,sdRAN_50_3_4_unbal)
#texstring += '\nFraction of juries with at least 1 \t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' % (ng1SAR_50_1_5_unbal,ng1STR_50_1_5_unbal,ng1SAR_50_2_4_unbal,ng1STR_50_2_4_unbal,ng1SAR_50_3_4_unbal,ng1STR_50_3_4_unbal,ng1RAN_50_3_4_unbal)
texstring += table_post
texstring += "\n\n"

texstring += table_pre
texstring += '     \\multicolumn{8}{c}{Slightly asymmetric$\n'

texstring += 'Average fraction of minorities \t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' %           (avSAR_50_1_5_tilt,avSTR_50_1_5_tilt,avSAR_50_2_4_tilt,avSTR_50_2_4_tilt,avSAR_50_3_4_tilt,avSTR_50_3_4_tilt,avRAN_50_3_4_tilt)
texstring += '\nStandard deviation  \t\t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' %     (sdSAR_50_1_5_tilt,sdSTR_50_1_5_tilt,sdSAR_50_2_4_tilt,sdSTR_50_2_4_tilt,sdSAR_50_3_4_tilt,sdSTR_50_3_4_tilt,sdRAN_50_3_4_tilt)
#texstring += '\nFraction of juries with at least 1 \t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' % (ng1SAR_50_1_5_tilt,ng1STR_50_1_5_tilt,ng1SAR_50_2_4_tilt,ng1STR_50_2_4_tilt,ng1SAR_50_3_4_tilt,ng1STR_50_3_4_tilt,ng1RAN_50_3_4_tilt)
texstring += table_post
print(texstring)

with open(imagedir + 'tables.tex', 'a') as outputfile:
    outputfile.write(texstring)
outputfile.close()


#%%
def equalsize_fig(savefig, values, v2, v3, filename, legend=True):
    
    # fig, (ax, ax2)= plt.subplots(1,2,figsize=(6,6), sharey=True, gridspec_kw={'width_ratios': [8, 1]})
    # plt.tight_layout()
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(6,4))
    # ax = fig.add_axes([0.,0.05,.64,1]) # axis starts at 0.1, 0.1
    # ax2 = fig.add_axes([0.6,0.05,.1,1])

    for ax in [ax1,ax2,ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_ylim(0,70)
        ax.set_yticks([])
        ax.tick_params(length=0)
    

    width = 0.4  # the width of the bars

    hatch3= ['..','\\\\','',]
    hatchrep = ['\\\\']*3
    hatchstr = ['..']*3

# draw average n. of minorities
    key = 0
    multiplier = 0
    offset = width * multiplier 
    rects = ax1.bar(np.arange(4) + offset, values['REP'], width, label=['REP']*4, color='white', hatch=hatchrep+[''], edgecolor = edgecolor)
    ax1.bar_label(rects, padding=3)
    multiplier += 1    

    key += 1 
    offset = width * multiplier 
    rects = ax1.bar(np.arange(3) + offset, values['STR'], width, label=['STR']*3, color='white', hatch=hatchstr, edgecolor = edgecolor)
    ax1.bar_label(rects, padding=3)
    multiplier += 1    

# draw average n. of minorities
    key = 0
    multiplier=0
    offset = width * multiplier 
    rects = ax2.bar(np.arange(4) + offset, v2['REP'], width, label=['REP']*4, color='white', hatch=hatchrep+[''], edgecolor = edgecolor)
    ax2.bar_label(rects, padding=3)
    multiplier += 1    

    key += 1 
    offset = width * multiplier
    rects = ax2.bar(np.arange(3) + offset, v2['STR'], width, label=['STR']*3, color='white', hatch=hatchstr, edgecolor = edgecolor)
    ax2.bar_label(rects, padding=3)
    multiplier += 1    

# draw average n. of minorities
    key = 0
    multiplier = 0
    offset = width * multiplier
    rects = ax3.bar(np.arange(4) + offset, v2['REP'], width, label=['REP']*4, color='white', hatch=hatchrep+[''], edgecolor = edgecolor)
    ax3.bar_label(rects, padding=3)
    multiplier += 1    

    key += 1 
    offset = width * multiplier 
    rects = ax3.bar(np.arange(3) + offset, v2['STR'], width, label=['STR']*3, color='white', hatch=hatchstr, edgecolor = edgecolor)
    ax3.bar_label(rects, padding=3)
    multiplier += 1   

    ax1.set_title('(a) $r=0.5$', fontsize=15)
    ax2.set_title('(b) $r=0.45$', fontsize=15)
    ax3.set_title('(c) $r=0.5, asymm.$', fontsize=15)

    ticklocks = [width/2, 1+width/2, 2+width/2, 3]
    ax1.set_xticks(ticklocks, ['Extr.','Mod.','Mild','(All)'])
    ax2.set_xticks(ticklocks, ['Extr.','Mod.','Mild','(All)'])
    ax3.set_xticks(ticklocks, [r'Extr.$^*$',r'Mod.$^*$',r'Mild$^*$','(All)'])
    ax.set_xlim(-width,width*8)


    handles, labels = ax1.get_legend_handles_labels()
    fig.legend([handles[0],handles[4],handles[3]],[rep_label,str_label,ran_label], loc='upper center', bbox_to_anchor=(0.5, 0.85), ncol=3, fontsize=15)
    
    fig.tight_layout()
    if savefig:
        plt.savefig(filename, pad_inches=.25, bbox_inches='tight')


# data from table 3
values = {'REP': [avSAR_50_b15,avSAR_50_b24,avSAR_50_b34, avRAN_50_b15 ], # [48, 49, 50],
          'STR': [avSTR_50_b15,avSTR_50_b24, avSTR_50_b34], #[50, 50, 50],
          'RAN': [avRAN_50_b15], #[50],
}
values1 = {'REP': [avSAR_50_1_5_unbal,avSAR_50_2_4_unbal,avSAR_50_3_4_unbal, avRAN_50_1_5_unbal], # [39, 42, 45],
           'STR': [avSTR_50_1_5_unbal,avSTR_50_2_4_unbal,avSTR_50_3_4_unbal], #[40, 42, 44],
           'RAN': [avRAN_50_1_5_unbal], #[45],
}
values2 = {'REP': [avSAR_50_1_5_tilt,avSAR_50_2_4_tilt,avSAR_50_3_4_tilt, avRAN_50_1_5_tilt], # [47, 49, 49],
           'STR': [avSTR_50_1_5_tilt,avSTR_50_2_4_tilt,avSTR_50_3_4_tilt], #[50, 48, 48],
           'RAN': [avRAN_50_1_5_tilt], #[50],
}

for vals in [values,values1,values2]:
    for mod,val in vals.items():
        vals[mod]= [np.round(item*100) for item in val]

equalsize_fig(True, values, values1, values2, filename=imagedir+'balanced.pdf', legend=False)
#%% Proposition 1, appendix fig:prop1-uni (uniform distribution)

fname = gzip.open(outputdir+'uni-12j-6-6.pickle.gz','rb')
resultuni = pickle.load(fname)
baseargs = resultuni['baseargs']
njuries = len(resultuni['juriestSAR'][:,0])
fname.close()

xSAR = resultuni['juriesxSAR']
xSTR = resultuni['juriesxSTR']
xRAN = resultuni['juriesxRAN']
nLess_c_SAR = np.array([])
nLess_c_STR = np.array([])
nLess_c_RAN = np.array([])
nMore_c_SAR = np.array([])
nMore_c_STR = np.array([])
nMore_c_RAN = np.array([])


xx = np.arange(0, 1.01, 0.01)
for c in xx:
    nLess_c_SAR = np.append(nLess_c_SAR, np.sum(np.count_nonzero(xSAR<c, axis=1) >=1 ) /njuries)
    nLess_c_STR = np.append(nLess_c_STR, np.sum(np.count_nonzero(xSTR<c, axis=1) >=1 ) /njuries)
    nLess_c_RAN = np.append(nLess_c_RAN, np.sum(np.count_nonzero(xRAN<c, axis=1) >=1 ) /njuries)
    nMore_c_SAR = np.append(nMore_c_SAR, np.sum(np.count_nonzero(xSAR>c, axis=1) >=1 ) /njuries)
    nMore_c_STR = np.append(nMore_c_STR, np.sum(np.count_nonzero(xSTR>c, axis=1) >=1 ) /njuries)
    nMore_c_RAN = np.append(nMore_c_RAN, np.sum(np.count_nonzero(xRAN>c, axis=1) >=1 ) /njuries)

fig,(ax) = plt.subplots(1,1,figsize=(4,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlim((0,.45))
ax.set_xlabel('Conviction probability $c$')
ax.set_ylabel('Fraction of juries')

ax.plot(xx, nLess_c_RAN, ':', color=RANcolor, linewidth=2, label=ran_label)
ax.plot(xx, nLess_c_SAR, '', color=SARcolor, label=rep_label)
ax.plot(xx, nLess_c_STR, '--', color=STRcolor, label=str_label)
ax.legend(loc="lower right")
#ax2.legend(loc="center left")
fig.tight_layout()

plt.savefig(imagedir+'uni-12-6-6.pdf')

#%% Appendix Table tab-app:betas-grouprep,
# minority representation when minority favor conviction

#note: we are drawing here the same distributions as in the main
#with T1 being now the minority, therefore statistics are computed
#averaging T1 jurors
fname = gzip.open(outputdir+'beta-1-5-12j-50pcT1.pickle.gz','rb')
resultbeta_50 = pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-1-5-12j-25pcT1.pickle.gz','rb')
resultbeta_25 =pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-1-5-12j-10pcT1.pickle.gz','rb')
resultbeta_10 = pickle.load(fname)
fname.close()

#compute statistics of  number type-1 of jurors (they are the minority
#and they favor conviction)
avSAR_50_b15 = np.average(np.average(resultbeta_50['juriestSAR'],axis=1))
avSTR_50_b15 = np.average(np.average(resultbeta_50['juriestSTR'],axis=1))
avRAN_50_b15 = np.average(np.average(resultbeta_50['juriestRAN'],axis=1))
sdSAR_50_b15 = np.std(np.average(resultbeta_50['juriestSAR'],axis=1))
sdSTR_50_b15 = np.std(np.average(resultbeta_50['juriestSTR'],axis=1))
sdRAN_50_b15 = np.std(np.average(resultbeta_50['juriestRAN'],axis=1))
ng1SAR_50_b15 = np.average(np.sum(resultbeta_50['juriestSAR'],axis=1)>=1)
ng1STR_50_b15 = np.average(np.sum(resultbeta_50['juriestSTR'],axis=1)>=1)
ng1RAN_50_b15 = np.average(np.sum(resultbeta_50['juriestRAN'],axis=1)>=1)

avSAR_25_b15 =np.average(np.average(resultbeta_25['juriestSAR'],axis=1))
avSTR_25_b15 =np.average(np.average(resultbeta_25['juriestSTR'],axis=1))
avRAN_25_b15 =np.average(np.average(resultbeta_25['juriestRAN'],axis=1))
sdSAR_25_b15 =np.std(np.average(resultbeta_25['juriestSAR'],axis=1))
sdSTR_25_b15 =np.std(np.average(resultbeta_25['juriestSTR'],axis=1))
sdRAN_25_b15 =np.std(np.average(resultbeta_25['juriestRAN'],axis=1))
ng1SAR_25_b15 =np.average(np.sum(resultbeta_25['juriestSAR'],axis=1)>=1)
ng1STR_25_b15 =np.average(np.sum(resultbeta_25['juriestSTR'],axis=1)>=1)
ng1RAN_25_b15 =np.average(np.sum(resultbeta_25['juriestRAN'],axis=1)>=1)

avSAR_10_b15 = np.average(np.average(resultbeta_10['juriestSAR'],axis=1))
avSTR_10_b15 = np.average(np.average(resultbeta_10['juriestSTR'],axis=1))
avRAN_10_b15 = np.average(np.average(resultbeta_10['juriestRAN'],axis=1))
sdSAR_10_b15 = np.std(np.average(resultbeta_10['juriestSAR'],axis=1))
sdSTR_10_b15 = np.std(np.average(resultbeta_10['juriestSTR'],axis=1))
sdRAN_10_b15 = np.std(np.average(resultbeta_10['juriestRAN'],axis=1))
ng1SAR_10_b15 = np.average(np.sum(resultbeta_10['juriestSAR'],axis=1)>=1)
ng1STR_10_b15 = np.average(np.sum(resultbeta_10['juriestSTR'],axis=1)>=1)
ng1RAN_10_b15 = np.average(np.sum(resultbeta_10['juriestRAN'],axis=1)>=1)

fname = gzip.open(outputdir+'beta-2-4-12j-50pcT1.pickle.gz','rb')
resultbeta_50 = pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-2-4-12j-25pcT1.pickle.gz','rb')
resultbeta_25 =pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-2-4-12j-10pcT1.pickle.gz','rb')
resultbeta_10 = pickle.load(fname)
fname.close()

#compute statistics of number of type-1 jurors(they are the minority
#and they favor conviction)
avSAR_50_b24 = np.average(np.average(resultbeta_50['juriestSAR'],axis=1))
avSTR_50_b24 = np.average(np.average(resultbeta_50['juriestSTR'],axis=1))
avRAN_50_b24 = np.average(np.average(resultbeta_50['juriestRAN'],axis=1))
sdSAR_50_b24 = np.std(np.average(resultbeta_50['juriestSAR'],axis=1))
sdSTR_50_b24 = np.std(np.average(resultbeta_50['juriestSTR'],axis=1))
sdRAN_50_b24 = np.std(np.average(resultbeta_50['juriestRAN'],axis=1))
ng1SAR_50_b24 = np.average(np.sum(resultbeta_50['juriestSAR'],axis=1)>=1)
ng1STR_50_b24 = np.average(np.sum(resultbeta_50['juriestSTR'],axis=1)>=1)
ng1RAN_50_b24 = np.average(np.sum(resultbeta_50['juriestRAN'],axis=1)>=1)

avSAR_25_b24 =np.average(np.average(resultbeta_25['juriestSAR'],axis=1))
avSTR_25_b24 =np.average(np.average(resultbeta_25['juriestSTR'],axis=1))
avRAN_25_b24 =np.average(np.average(resultbeta_25['juriestRAN'],axis=1))
sdSAR_25_b24 =np.std(np.average(resultbeta_25['juriestSAR'],axis=1))
sdSTR_25_b24 =np.std(np.average(resultbeta_25['juriestSTR'],axis=1))
sdRAN_25_b24 =np.std(np.average(resultbeta_25['juriestRAN'],axis=1))
ng1SAR_25_b24 =np.average(np.sum(resultbeta_25['juriestSAR'],axis=1)>=1)
ng1STR_25_b24 =np.average(np.sum(resultbeta_25['juriestSTR'],axis=1)>=1)
ng1RAN_25_b24 =np.average(np.sum(resultbeta_25['juriestRAN'],axis=1)>=1)

avSAR_10_b24 = np.average(np.average(resultbeta_10['juriestSAR'],axis=1))
avSTR_10_b24 = np.average(np.average(resultbeta_10['juriestSTR'],axis=1))
avRAN_10_b24 = np.average(np.average(resultbeta_10['juriestRAN'],axis=1))
sdSAR_10_b24 = np.std(np.average(resultbeta_10['juriestSAR'],axis=1))
sdSTR_10_b24 = np.std(np.average(resultbeta_10['juriestSTR'],axis=1))
sdRAN_10_b24 = np.std(np.average(resultbeta_10['juriestRAN'],axis=1))
ng1SAR_10_b24 = np.average(np.sum(resultbeta_10['juriestSAR'],axis=1)>=1)
ng1STR_10_b24 = np.average(np.sum(resultbeta_10['juriestSTR'],axis=1)>=1)
ng1RAN_10_b24 = np.average(np.sum(resultbeta_10['juriestRAN'],axis=1)>=1)

fname = gzip.open(outputdir+'beta-3-4-12j-50pcT1.pickle.gz','rb')
resultbeta_50 = pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-3-4-12j-25pcT1.pickle.gz','rb')
resultbeta_25 =pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-3-4-12j-10pcT1.pickle.gz','rb')
resultbeta_10 = pickle.load(fname)
fname.close()

#compute statistics of  number type-1 of jurors (they are the minority
#and they favor conviction)
avSAR_50_b34 = np.average(np.average(resultbeta_50['juriestSAR'],axis=1))
avSTR_50_b34 = np.average(np.average(resultbeta_50['juriestSTR'],axis=1))
avRAN_50_b34 = np.average(np.average(resultbeta_50['juriestRAN'],axis=1))
sdSAR_50_b34 = np.std(np.average(resultbeta_50['juriestSAR'],axis=1))
sdSTR_50_b34 = np.std(np.average(resultbeta_50['juriestSTR'],axis=1))
sdRAN_50_b34 = np.std(np.average(resultbeta_50['juriestRAN'],axis=1))
ng1SAR_50_b34 = np.average(np.sum(resultbeta_50['juriestSAR'],axis=1)>=1)
ng1STR_50_b34 = np.average(np.sum(resultbeta_50['juriestSTR'],axis=1)>=1)
ng1RAN_50_b34 = np.average(np.sum(resultbeta_50['juriestRAN'],axis=1)>=1)

avSAR_25_b34 =np.average(np.average(resultbeta_25['juriestSAR'],axis=1))
avSTR_25_b34 =np.average(np.average(resultbeta_25['juriestSTR'],axis=1))
avRAN_25_b34 =np.average(np.average(resultbeta_25['juriestRAN'],axis=1))
sdSAR_25_b34 =np.std(np.average(resultbeta_25['juriestSAR'],axis=1))
sdSTR_25_b34 =np.std(np.average(resultbeta_25['juriestSTR'],axis=1))
sdRAN_25_b34 =np.std(np.average(resultbeta_25['juriestRAN'],axis=1))
ng1SAR_25_b34 =np.average(np.sum(resultbeta_25['juriestSAR'],axis=1)>=1)
ng1STR_25_b34 =np.average(np.sum(resultbeta_25['juriestSTR'],axis=1)>=1)
ng1RAN_25_b34 =np.average(np.sum(resultbeta_25['juriestRAN'],axis=1)>=1)

avSAR_10_b34 = np.average(np.average(resultbeta_10['juriestSAR'],axis=1))
avSTR_10_b34 = np.average(np.average(resultbeta_10['juriestSTR'],axis=1))
avRAN_10_b34 = np.average(np.average(resultbeta_10['juriestRAN'],axis=1))
sdSAR_10_b34 = np.std(np.average(resultbeta_10['juriestSAR'],axis=1))
sdSTR_10_b34 = np.std(np.average(resultbeta_10['juriestSTR'],axis=1))
sdRAN_10_b34 = np.std(np.average(resultbeta_10['juriestRAN'],axis=1))
ng1SAR_10_b34 = np.average(np.sum(resultbeta_10['juriestSAR'],axis=1)>=1)
ng1STR_10_b34 = np.average(np.sum(resultbeta_10['juriestSTR'],axis=1)>=1)
ng1RAN_10_b34 = np.average(np.sum(resultbeta_10['juriestRAN'],axis=1)>=1)


texstring = '\n\n Appendix Table tab-app:betas-grouprep \n\n'
texstring += 'Average fraction of minorities \t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' %           (avSAR_25_b15,avSTR_25_b15,avSAR_25_b24,avSTR_25_b24,avSAR_25_b34,avSTR_25_b34,avRAN_25_b34)
texstring += '\nStandard deviation  \t\t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' %     (sdSAR_25_b15,sdSTR_25_b15,sdSAR_25_b24,sdSTR_25_b24,sdSAR_25_b34,sdSTR_25_b34,sdRAN_25_b34)
texstring += '\nFraction of juries with at least 1 \t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' % (ng1SAR_25_b15,ng1STR_25_b15,ng1SAR_25_b24,ng1STR_25_b24,ng1SAR_25_b34,ng1STR_25_b34,ng1RAN_25_b34)

texstring += '\n\nAverage fraction of minorities \t & \t %3.2f  \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' %           (avSAR_10_b15,avSTR_10_b15,avSAR_10_b24,avSTR_10_b24,avSAR_10_b34,avSTR_10_b34,avRAN_10_b34)
texstring += '\nStandard deviation  \t\t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' %     (sdSAR_10_b15,sdSTR_10_b15,sdSAR_10_b24,sdSTR_10_b24,sdSAR_10_b34,sdSTR_10_b34,sdRAN_10_b34)
texstring += '\nFraction of juries with at least 1 \t & \t %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \t& %3.2f \\\\' % (ng1SAR_10_b15,ng1STR_10_b15,ng1SAR_10_b24,ng1STR_10_b24,ng1SAR_10_b34,ng1STR_10_b34,ng1RAN_10_b34)
print(texstring)

with open(imagedir + 'tables.tex', 'a') as outputfile:
    outputfile.write(texstring)
outputfile.close()

#%% Appendix fig:prop1-uniProp 4(a), (jurors above median, less polarized distros)

# fig:median-2-4
fname = gzip.open(outputdir+'beta-2-4-12j-50pcT1.pickle.gz','rb')
resultbeta_50 = pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-2-4-12j-75pcT1.pickle.gz','rb')
resultbeta_75 = pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-2-4-12j-90pcT1.pickle.gz','rb')
resultbeta_90 = pickle.load(fname)
fname.close()

#get median
median50 = np.median(resultbeta_50['juriesxRAN'])
median75 = np.median(resultbeta_75['juriesxRAN'])
median90 = np.median(resultbeta_90['juriesxRAN'])

# compute prob that n jurors are above mediand
nSARgmed_50 = np.sum(resultbeta_50['juriesxSAR']<median50,axis=1)
nSARgmed_75 = np.sum(resultbeta_75['juriesxSAR']<median75,axis=1)
nSARgmed_90 = np.sum(resultbeta_90['juriesxSAR']<median90,axis=1)
nSTRgmed_50 = np.sum(resultbeta_50['juriesxSTR']<median50,axis=1)
nSTRgmed_75 = np.sum(resultbeta_75['juriesxSTR']<median75,axis=1)
nSTRgmed_90 = np.sum(resultbeta_90['juriesxSTR']<median90,axis=1)
nRANgmed_90 = np.sum(resultbeta_90['juriesxRAN']<median90,axis=1)
nRANgmed_50 = np.sum(resultbeta_50['juriesxRAN']<median50,axis=1)
nRANgmed_75 = np.sum(resultbeta_75['juriesxRAN']<median75,axis=1)

njurors = resultbeta_50['juriesxRAN'].shape[1]
prgmed50SAR = np.zeros(njurors)
prgmed75SAR = np.zeros(njurors)
prgmed90SAR = np.zeros(njurors)
prgmed50STR = np.zeros(njurors)
prgmed75STR = np.zeros(njurors)
prgmed90STR = np.zeros(njurors)
prgmed50RAN = np.zeros(njurors)
prgmed75RAN = np.zeros(njurors)
prgmed90RAN = np.zeros(njurors)

for n in range(6,13):
    prgmed50SAR[n-1] = np.sum(nSARgmed_50>=n)/len(nSARgmed_50)
    prgmed75SAR[n-1] = np.sum(nSARgmed_75>=n)/len(nSARgmed_75)
    prgmed90SAR[n-1] = np.sum(nSARgmed_90>=n)/len(nSARgmed_90)
    prgmed50STR[n-1] = np.sum(nSTRgmed_50>=n)/len(nSTRgmed_50)
    prgmed75STR[n-1] = np.sum(nSTRgmed_75>=n)/len(nSTRgmed_75)
    prgmed90STR[n-1] = np.sum(nSTRgmed_90>=n)/len(nSTRgmed_90)
    prgmed50RAN[n-1] = np.sum(nRANgmed_50>=n)/len(nRANgmed_50)
    prgmed75RAN[n-1] = np.sum(nRANgmed_75>=n)/len(nRANgmed_75)
    prgmed90RAN[n-1] = np.sum(nRANgmed_90>=n)/len(nRANgmed_90)

print('beta 2-4')
print('medians [50,75,90]:',median50,median75,median90)
print('SAR, 50%T1', prgmed50SAR[6:12])
print('STR, 50%T1', prgmed50STR[6:12])
print('SAR, 75%T1', prgmed75SAR[6:12])
print('STR, 75%T1', prgmed75STR[6:12])
print('SAR, 90%T1', prgmed90SAR[6:12])
print('STR, 90%T1', prgmed90STR[6:12])

x = np.arange(7,13)

fig,ax = plt.subplots(figsize=(4,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(6.8,12.7)
ax.set_ylim(-0.015,0.099)
ax.set_xlabel('Number of jurors')
ax.set_ylabel('Probability (difference with \\textit{RAN})')
shift=.2

ax.plot(x,prgmed50STR[6:13]-prgmed50RAN[6:13],'--',marker='*',color=STRcolor,label=str_label,linewidth=.8)
ax.plot(x,prgmed90SAR[6:13]-prgmed90RAN[6:13],marker='s',color=RANcolor,label=r90,linewidth=.8)
ax.plot(x,prgmed75SAR[6:13]-prgmed75RAN[6:13],':',marker='o',color=SARcolor,label=r75,linewidth=.8)
ax.plot(x,prgmed50SAR[6:13]-prgmed50RAN[6:13],'-.',marker='^',color='goldenrod',label=r50,linewidth=.8)
fig.tight_layout()
ax.legend()
plt.savefig(imagedir+'median-2-4.pdf')

#%% Appendix fig:prop1-uniProp 4(b), (jurors above median, less polarized distros)

# fig:median-3-4
fname = gzip.open(outputdir+'beta-3-4-12j-50pcT1.pickle.gz','rb')
resultbeta_50 = pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-3-4-12j-75pcT1.pickle.gz','rb')
resultbeta_75 = pickle.load(fname)
fname.close()
fname = gzip.open(outputdir+'beta-3-4-12j-90pcT1.pickle.gz','rb')
resultbeta_90 = pickle.load(fname)
fname.close()

#resultbeta_50['juriestRAN'],axis=1)
median50 = np.median(resultbeta_50['juriesxRAN'])
median75 = np.median(resultbeta_75['juriesxRAN'])
median90 = np.median(resultbeta_90['juriesxRAN'])

# compute prob that n jurors are above median
nSARgmed_50 = np.sum(resultbeta_50['juriesxSAR']<median50,axis=1)
nSARgmed_75 = np.sum(resultbeta_75['juriesxSAR']<median75,axis=1)
nSARgmed_90 = np.sum(resultbeta_90['juriesxSAR']<median90,axis=1)
nSTRgmed_50 = np.sum(resultbeta_50['juriesxSTR']<median50,axis=1)
nSTRgmed_75 = np.sum(resultbeta_75['juriesxSTR']<median75,axis=1)
nSTRgmed_90 = np.sum(resultbeta_90['juriesxSTR']<median90,axis=1)
nRANgmed_90 = np.sum(resultbeta_90['juriesxRAN']<median90,axis=1)
nRANgmed_50 = np.sum(resultbeta_50['juriesxRAN']<median50,axis=1)
nRANgmed_75 = np.sum(resultbeta_75['juriesxRAN']<median75,axis=1)

njurors = resultbeta_50['juriesxRAN'].shape[1]
prgmed50SAR = np.zeros(njurors)
prgmed75SAR = np.zeros(njurors)
prgmed90SAR = np.zeros(njurors)
prgmed50STR = np.zeros(njurors)
prgmed75STR = np.zeros(njurors)
prgmed90STR = np.zeros(njurors)
prgmed50RAN = np.zeros(njurors)
prgmed75RAN = np.zeros(njurors)
prgmed90RAN = np.zeros(njurors)
for n in range(6,13):
    prgmed50SAR[n-1] = np.sum(nSARgmed_50>=n)/len(nSARgmed_50)
    prgmed75SAR[n-1] = np.sum(nSARgmed_75>=n)/len(nSARgmed_75)
    prgmed90SAR[n-1] = np.sum(nSARgmed_90>=n)/len(nSARgmed_90)
    prgmed50STR[n-1] = np.sum(nSTRgmed_50>=n)/len(nSTRgmed_50)
    prgmed75STR[n-1] = np.sum(nSTRgmed_75>=n)/len(nSTRgmed_75)
    prgmed90STR[n-1] = np.sum(nSTRgmed_90>=n)/len(nSTRgmed_90)
    prgmed50RAN[n-1] = np.sum(nRANgmed_50>=n)/len(nRANgmed_50)
    prgmed75RAN[n-1] = np.sum(nRANgmed_75>=n)/len(nRANgmed_75)
    prgmed90RAN[n-1] = np.sum(nRANgmed_90>=n)/len(nRANgmed_90)

print('beta 3-4')
print('medians [50,75,90]:',median50,median75,median90)
print('SAR, 50%T1', prgmed50SAR[6:12])
print('STR, 50%T1', prgmed50STR[6:12])
print('SAR, 75%T1', prgmed75SAR[6:12])
print('STR, 75%T1', prgmed75STR[6:12])
print('SAR, 90%T1', prgmed90SAR[6:12])
print('STR, 90%T1', prgmed90STR[6:12])

x = np.arange(7,13)

fig,ax = plt.subplots(figsize=(4,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(6.8,12.7)
ax.set_ylim(-0.015,0.099)
ax.set_xlabel('Number of jurors')
ax.set_ylabel('Probability (difference with \\textit{RAN})')
shift=.2

ax.plot(x,prgmed50STR[6:13]-prgmed50RAN[6:13],'--',marker='*',color=STRcolor,label=str_label,linewidth=.8)
ax.plot(x,prgmed90SAR[6:13]-prgmed90RAN[6:13],marker='s',color=RANcolor,label=r90,linewidth=.8)
ax.plot(x,prgmed75SAR[6:13]-prgmed75RAN[6:13],':',marker='o',color=SARcolor,label=r75,linewidth=.8)
ax.plot(x,prgmed50SAR[6:13]-prgmed50RAN[6:13],'-.',marker='^',color='goldenrod',label=r50,linewidth=.8)
fig.tight_layout()
ax.legend()
plt.savefig(imagedir+'median-3-4.pdf')

# %%
