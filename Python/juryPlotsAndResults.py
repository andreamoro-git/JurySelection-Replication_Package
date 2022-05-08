#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# In this file we draw figures and compute stats. Juries are generated in
# file juryConstruction.py
#
# (note: let's try to keep only those actually used in the paper)

#%% Module import (run before running any cell)

import gzip,pickle
import matplotlib.pyplot as plt
import numpy as np

# erasae if this generates errors
import matplotlib
import labellines

imagedir = 'exhibits/'
outputdir = 'output/'

# this can be commented out if your system does not have or cannot
# read a LaTeX distribution.
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

# import os
# os.environ['PATH'] += '/Library/TeX/texbin'

table_pre = '\n\\begin{tabular}{lccccccc}'
table_pre+= 	'\n	 Polarization &\\multicolumn{2}{c}{Extreme}'
table_pre+= 	'\n 	 &\\multicolumn{2}{c}{Moderate}'
table_pre+= 	'\n 	 &\\multicolumn{2}{c}{Mild}'
table_pre+= 	'\n 	 & (All) \\\\'
table_pre+= 	'\n 	 Procedure    & \\SAR & \\STR  & \\SAR & \\STR  & \\SAR & \\STR  & \\RAN \\\\'
table_pre+= 	'\n 	 \\hline\n'
table_post = '\n\\end{tabular}\n'

#%% Plot beta densities (fig:betaPDFs)

from math import gamma

x = np.linspace(0,1,100)
alpha = 1
beta = 5
ya = np.multiply(x**(alpha-1),(1-x)**(beta-1)) * gamma(alpha+beta)/(gamma(alpha)*gamma(beta))

alpha = 5
beta = 1
yb = np.multiply(x**(alpha-1),(1-x)**(beta-1)) * gamma(alpha+beta)/(gamma(alpha)*gamma(beta))

#yb = np.multiply(x**(alpha-1),(1-x)**(beta-1)) /( (alpha+beta)/(alpha*beta)/nCr(alpha+beta,alpha) )

fig,ax = plt.subplots(figsize=(2.2,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(-0.01,1.01)
ax.set_xlabel('Conviction probability')

plt.plot(x,ya,color='Olive',label='$f_a(c)$: $Beta(1,5)$')
plt.plot(x,yb,'--',color='DarkOrange',label='$f_b(c)$: $Beta(5,1)$')
ax.legend(loc='center')
fig.tight_layout()

plt.savefig(imagedir+'beta_1-5_dist.pdf')

alpha = 2
beta = 4
ya = np.multiply(x**(alpha-1),(1-x)**(beta-1)) * gamma(alpha+beta)/(gamma(alpha)*gamma(beta))
alpha = 4
beta = 2
yb = np.multiply(x**(alpha-1),(1-x)**(beta-1)) * gamma(alpha+beta)/(gamma(alpha)*gamma(beta))

fig,ax = plt.subplots(figsize=(2.2,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(-0.01,1.01)
ax.set_xlabel('Conviction probability')

plt.plot(x,ya,color='Olive',label='$f_a(c)$: $Beta(2,4)$')
plt.plot(x,yb,'--',color='DarkOrange',label='$f_b(c)$: $Beta(4,2)$')
ax.legend(loc='center')
fig.tight_layout()

plt.savefig(imagedir+'beta_2-4_dist.pdf')

alpha = 3
beta = 4
ya = np.multiply(x**(alpha-1),(1-x)**(beta-1)) * gamma(alpha+beta)/(gamma(alpha)*gamma(beta))
alpha = 4
beta = 3
yb = np.multiply(x**(alpha-1),(1-x)**(beta-1)) * gamma(alpha+beta)/(gamma(alpha)*gamma(beta))

fig,ax = plt.subplots(figsize=(2.2,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(-0.01,1.01)
ax.set_xlabel('Conviction probability $c$')

plt.plot(x,ya,color='Olive',label='$f_a(c)$: $Beta(3,4)$')
plt.plot(x,yb,'--',color='DarkOrange',label='$f_b(c)$: $Beta(4,3)$')
# from labellines import labelLines
# plt.gca().get_lines()
# labelLines(plt.gca().get_lines(), align=False, xvals= [0.25,0.82], fontsize=11)

ax.legend(loc='center')
fig.tight_layout()

plt.savefig(imagedir+'beta_3-4_dist.pdf')


#%% fig:atleast1-3betas (for proposition 1)

## extreme polarization
fname = gzip.open(outputdir+'beta-1-5-12j-75pcT1.pickle.gz','rb')
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

fig,(ax) = plt.subplots(1,1,figsize=(2.2,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlim((0,.45))
ax.set_xlabel('Threshold $\\underline{c}$')
ax.set_ylabel('Fraction of juries')

ax.plot(xx, nLess_c_RAN, ':', color='saddlebrown', linewidth=2, label='\\textit{RAN}')
ax.plot(xx, nLess_c_SAR, '', color='DarkOrange', label='\\textit{REP')
ax.plot(xx, nLess_c_STR, '--', color='Olive', label='\\textit{STR}')
#ax.legend(loc="lower right")
labellines.labelLines(plt.gca().get_lines(), align=False, fontsize=11,
                      xvals = [.1,.25,.4],
                      yoffsets=[.18,.07, -.1],
                      bbox={'alpha': 0},
                            )
fig.tight_layout()

plt.savefig(imagedir+'prop1-beta-1-5-75pcT1-ul.pdf')

# Numbers in text
textstring = ''
textstring += 'Extreme, 10th pctile='+str(np.percentile(xRAN,10))
textstring += '\n verify: '+str(xx[10:11])
textstring += '\n STR' +str(nLess_c_STR[10:11])
textstring += '\n SAR' +str(nLess_c_SAR[10:11])
textstring += '\n RAN' +str(nLess_c_RAN[10:11])
#%%
## moderate polarization
fname = gzip.open(outputdir+'beta-2-4-12j-75pcT1.pickle.gz','rb')
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

fig,(ax) = plt.subplots(1,1,figsize=(2.2,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlim((0,.45))
ax.set_xlabel('Threshold $\\underline{c}$')
ax.set_ylabel('Fraction of juries')

ax.plot(xx, nLess_c_RAN, ':', color='saddlebrown', linewidth=2, label='\\textit{RAN}')
ax.plot(xx, nLess_c_SAR, '', color='DarkOrange', label='\\textit{REP')
ax.plot(xx, nLess_c_STR, '--', color='Olive', label='\\textit{STR}')
#ax.legend(loc="lower right")
labellines.labelLines(plt.gca().get_lines(), align=False, fontsize=11,
                      xvals = [.09,.24,.4],
                      yoffsets=[.26,.12, -.22],
                      bbox={'alpha': 0},
                            )
fig.tight_layout()

plt.savefig(imagedir+'prop1-beta-2-4-75pcT1-ul.pdf')

# Numbers in text
textstring += '\n\nModerate, 10th pctile' + str(np.percentile(xRAN,10))
textstring += '\nverify: '+str(xx[24:25])
textstring += '\n STR ' +str(nLess_c_STR[24:25])
textstring += '\n SAR ' +str(nLess_c_SAR[24:25])
textstring += '\n RAN ' +str(nLess_c_RAN[24:25])

#%% mild polarization
fname = gzip.open(outputdir+'beta-3-4-12j-75pcT1.pickle.gz','rb')
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


fig,(ax) = plt.subplots(1,1,figsize=(2.2,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlim((0,.45))
ax.set_xlabel('Threshold $\\underline{c}$')
ax.set_ylabel('Fraction of juries')

ax.plot(xx, nLess_c_RAN, ':', color='saddlebrown', linewidth=2, label='\\textit{RAN}')
ax.plot(xx, nLess_c_SAR, '', color='DarkOrange', label='\\textit{REP')
ax.plot(xx, nLess_c_STR, '--', color='Olive', label='\\textit{STR}')
#ax.legend(loc="upper left")
#ax2.legend(loc="center left")
fig.tight_layout()
labellines.labelLines(plt.gca().get_lines(), align=False, fontsize=11,
                      xvals = [.12,.28,.4],
                      yoffsets=[.2,.15, -.3],
                      bbox={'alpha': 0},
                            )

plt.savefig(imagedir+'prop1-beta-3-4-75pcT1-ul.pdf')

# Numbers in text
textstring += '\n\nMild 10th pctile=' + str(np.percentile(xRAN,10))
textstring += '\nverify: '+str(xx[28:29])
textstring += '\n STR' + str(nLess_c_STR[28:29])
textstring += '\n SAR' + str(nLess_c_SAR[28:29])
textstring += '\n RAN' + str(nLess_c_RAN[28:29])

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

fig,(ax) = plt.subplots(1,1,figsize=(4,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlim((0,0.1))
ax.set_ylim((0,0.9))
ax.set_xlabel('Threshold $\\underline{c}$')
ax.set_ylabel('Fraction of juries')

ax.plot(xx, nLess_c_RAN, ':', color='saddlebrown', linewidth=2, label='\\textit{RAN}')
ax.plot(xx, nLess_c_SAR, '', color='DarkOrange', label='\\textit{REP')
ax.plot(xx, nLess_c_STR, '--', color='Olive', label='\\textit{STR}')
#ax.legend(loc="center left")
labellines.labelLines(plt.gca().get_lines(), align=False, fontsize=12,
                      xvals = [.09,.02,.04],
                      yoffsets=[-.08,0.09, -.08],
                      bbox={'alpha': 0},
                            )

fig.tight_layout()

plt.savefig(imagedir+'prop2-uni.pdf')

#%% Minority representation in size-1 juries (fig:counter)

fig,ax = plt.subplots(figsize=(3,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(-0.01,1.01)
ax.set_xlabel('Conviction probability $c$')
ax.set_yticks(np.arange(0,2,0.5))
xx = np.linspace(0,1,1000)
SAR = np.ones(len(xx))*.75
SAR[374:625]=1.75
RAN =  np.ones(len(xx))

STR = 6*xx*(1-xx)

ax.set_ylabel('Density')
ax.set_xlabel('Conviction probability $c$')

ax.plot(xx, RAN, ':', color='saddlebrown', linewidth=2, label='\\textit{RAN}')
ax.plot(xx, SAR, '', color='DarkOrange', label='\\textit{REP}')
ax.plot(xx, STR, '--', color='Olive', label='\\textit{STR}')

#ax.legend()
labellines.labelLines(plt.gca().get_lines(), align=False, fontsize=12,
                      xvals = [.9,.3,.2],
                      yoffsets=[.1,-0.12,.32],
                      bbox={'alpha': 0},
                            )

fig.tight_layout()
plt.savefig(imagedir+'counter.pdf')

#%% Minority representation in size-1 juries (fig:counter_b)

import matplotlib
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

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


fig,ax = plt.subplots(figsize=(3,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(0,.42)
ax.set_ylabel('Fraction of group-a in juries')
ax.set_xlabel('Fraction of group-a in jury pool $r$')
ax.set_yticks(np.arange(0,0.41,0.1))
ax.set_xticks(np.arange(0,0.41,0.1))

ax.plot(Rvector, avRAN, ':', color='saddlebrown', linewidth=2, label='\\textit{RAN}')
ax.plot(Rvector, avSAR, '', color='DarkOrange', label='\\textit{REP}')
ax.plot(Rvector, avSTR, '--', color='Olive', label='\\textit{STR}')

fig.tight_layout()
labellines.labelLines(plt.gca().get_lines(), align=False, fontsize=12,
                      xvals = [.1,.36,.2],
                      yoffsets=[.05,-0.045, -.05],
                      bbox={'alpha': 0},
                            )
#ax.legend()

plt.savefig(imagedir+'counter_b.pdf')

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

    # now draw the figures
    # the following package requires pip install matplotlib-label-lines
    import labellines

    fig,ax = plt.subplots(1,1,figsize=(3,3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(0,len(challenges)+1,2))
    ax.set_xlabel('Number of challenges')
    ax.set_ylabel('Fraction of juries')
    ax.plot(challenges, nLess_c_SAR, marker='s', color='darkorange', label='\\textit{REP}', linewidth=.6, markersize=4)
    ax.plot(challenges, nLess_c_STR, '--', marker='*',color='Olive', label='\\textit{STR}', linewidth=.6, markersize=4)
    #ax.legend(loc='center right')
    labellines.labelLines(plt.gca().get_lines(), align=False, yoffsets=[.06,.03], bbox={'alpha': 0}, fontsize=13, )# align=False)
    fig.tight_layout()
    plt.savefig(imagedir+'nchallenges-extreme.pdf')


    fig,ax = plt.subplots(1,1,figsize=(3,3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(0,len(challenges)+1,2))
    ax.set_xlabel('Number of challenges')
    ax.set_ylabel('Fraction of minorities')
    ax.plot(challenges, minSAR, marker='s', color='darkorange', label='\\textit{REP}', linewidth=.6, markersize=4)
    ax.plot(challenges, minSTR, '--', marker='*',color='Olive', label='\\textit{STR}', linewidth=.6, markersize=4)
    #ax.legend(loc='center right')
    labellines.labelLines(plt.gca().get_lines(), align=False, yoffsets=[.01,.006], bbox={'alpha': 0}, fontsize=13, )# align=False)
    fig.tight_layout()
    plt.savefig(imagedir+'nchallenges-minority.pdf')

#%% Prob. of selecting x jurors abovbe median (fig:median)
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
ax.set_xlabel('Minimum number of jurors below the median')
ax.set_ylabel('Fraction of juries (difference with \\textit{RAN})')
shift=.2

# ax.bar(x-2*shift,prgmed50STR[6:13]-prgmed50RAN[6:13],width=shift,color='Olive',label='\\textit{STR}')
# ax.bar(x-shift,prgmed90SAR[6:13]-prgmed90RAN[6:13],width=shift,color='SaddleBrown',label='\\textit{REP}, $r=.90$')
# ax.bar(x+0*shift,prgmed75SAR[6:13]-prgmed75RAN[6:13],width=shift,color='DarkOrange',label='\\textit{REP}, $r=.75$')
# ax.bar(x+1*shift,prgmed50SAR[6:13]-prgmed50RAN[6:13],width=shift,color='Gold',label='\\textit{REP}, $r=.5$')
ax.plot(x,prgmed50STR[6:13]-prgmed50RAN[6:13],'--',marker='*',color='Olive',label='\\textit{STR}',linewidth=.8)
ax.plot(x,prgmed90SAR[6:13]-prgmed90RAN[6:13],marker='s',color='SaddleBrown',label='\\parbox{5em}{\\textit{REP},\\newline $r=.10$}',linewidth=.8, markersize=4)
ax.plot(x,prgmed75SAR[6:13]-prgmed75RAN[6:13],':',marker='o',color='DarkOrange',label='\\textit{REP}, $r=.25$',linewidth=.8, markersize=4)
ax.plot(x,prgmed50SAR[6:13]-prgmed50RAN[6:13],'-.',marker='^',color='Goldenrod',label='\\textit{REP}, $r=.5$',linewidth=.8, markersize=4)
fig.tight_layout()
#ax.legend()
labellines.labelLines(plt.gca().get_lines(), align=False, fontsize=11,
                      xvals = [10.5,7.3,8.35,8.5],
                      yoffsets=[0.012,+0.019,-0.012,-0.016],
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

ax.plot(xx, nLess_c_RAN, ':', color='saddlebrown', linewidth=2, label='\\textit{RAN}')
ax.plot(xx, nLess_c_SAR, '', color='DarkOrange', label='\\textit{REP')
ax.plot(xx, nLess_c_STR, '--', color='Olive', label='\\textit{STR}')
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

# ax.bar(x-2*shift,prgmed50STR[6:13]-prgmed50RAN[6:13],width=shift,color='Olive',label='\\textit{STR}')
# ax.bar(x-shift,prgmed90SAR[6:13]-prgmed90RAN[6:13],width=shift,color='SaddleBrown',label='\\textit{REP}, $r=.90$')
# ax.bar(x+0*shift,prgmed75SAR[6:13]-prgmed75RAN[6:13],width=shift,color='DarkOrange',label='\\textit{REP}, $r=.75$')
# ax.bar(x+1*shift,prgmed50SAR[6:13]-prgmed50RAN[6:13],width=shift,color='Gold',label='\\textit{REP}, $r=.5$')
ax.plot(x,prgmed50STR[6:13]-prgmed50RAN[6:13],'--',marker='*',color='Olive',label='\\textit{STR}',linewidth=.8)
ax.plot(x,prgmed90SAR[6:13]-prgmed90RAN[6:13],marker='s',color='SaddleBrown',label='\\textit{REP}, $r=.90$',linewidth=.8)
ax.plot(x,prgmed75SAR[6:13]-prgmed75RAN[6:13],':',marker='o',color='DarkOrange',label='\\textit{REP}, $r=.75$',linewidth=.8)
ax.plot(x,prgmed50SAR[6:13]-prgmed50RAN[6:13],'-.',marker='^',color='goldenrod',label='\\textit{REP}, $r=.5$',linewidth=.8)
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

# ax.bar(x-2*shift,prgmed50STR[6:13]-prgmed50RAN[6:13],width=shift,color='Olive',label='\\textit{STR}')
# ax.bar(x-shift,prgmed90SAR[6:13]-prgmed90RAN[6:13],width=shift,color='SaddleBrown',label='\\textit{REP}, $r=.90$')
# ax.bar(x+0*shift,prgmed75SAR[6:13]-prgmed75RAN[6:13],width=shift,color='DarkOrange',label='\\textit{REP}, $r=.75$')
# ax.bar(x+1*shift,prgmed50SAR[6:13]-prgmed50RAN[6:13],width=shift,color='Gold',label='\\textit{REP}, $r=.5$')
ax.plot(x,prgmed50STR[6:13]-prgmed50RAN[6:13],'--',marker='*',color='Olive',label='\\textit{STR}',linewidth=.8)
ax.plot(x,prgmed90SAR[6:13]-prgmed90RAN[6:13],marker='s',color='SaddleBrown',label='\\textit{REP}, $r=.90$',linewidth=.8)
ax.plot(x,prgmed75SAR[6:13]-prgmed75RAN[6:13],':',marker='o',color='DarkOrange',label='\\textit{REP}, $r=.75$',linewidth=.8)
ax.plot(x,prgmed50SAR[6:13]-prgmed50RAN[6:13],'-.',marker='^',color='goldenrod',label='\\textit{REP}, $r=.5$',linewidth=.8)
fig.tight_layout()
ax.legend()
plt.savefig(imagedir+'median-3-4.pdf')
