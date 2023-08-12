#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 19:40:08 2023

@author: moroa
"""
#%%
import gzip,pickle,copy,shutil
from copy import deepcopy
from class_jury_Statdisc import Jury_statdisc, Jurypool
from class_model_types import Jurymodel
import pandas as pd, numpy as np
import labellines 
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

savefig = True

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
labelfont = 9
xylabelfont = 11

imagedir = '../Exhibits/'
outputdir = '../Simulations/'

# may need this to make sure LaTeX is found
import os
os.environ['PATH'] += '/Library/TeX/texbin'

# checks if your system does not have or cannot read a LaTeX distribution.
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
else:
    underline_c_label = '_c_'
    rep_label = 'REP'
    ran_label = 'RAN'
    str_label = 'STR'

# needed for debugging since I run this script from different folders
import os
if os.getcwd()[-7:] == 'Package':
    os.chdir(os.getcwd() + '/Code')

baseargs = {'J' : 12,'D' : 6, 'P' : 6, 
        'R' : 0.75,
        'fx0' : {'f': 'logit-normal', 'mu' : -1, 'sig': 1, 'lb': 0, 'ub': 1},
        'fx1' : {'f': 'logit-normal', 'mu' : 1,  'sig': 1, 'lb': 0, 'ub': 1},
        'print_option': 1,
        'delta' : 1e-4,
        'seed' : 2443,
        }

njuries = 50000

def computeStats(juries):
    data = []
    for i in range(len(juries)):
        data.append({'N': juries[i]['baseargs']['N'],
                            'fracSAR': np.average(np.sum(1-juries[i]['juriestSAR'],axis=1)>=1),
                            'fracSTR': np.average(np.sum(1-juries[i]['juriestSTR'],axis=1)>=1),
                            'fracRAN': np.average(np.sum(1-juries[i]['juriestRAN'],axis=1)>=1),
                            'diffFrac': np.average(np.sum(1-juries[i]['juriestSAR'],axis=1)>=1) - np.average(np.sum(1-juries[i]['juriestSTR'],axis=1)>=1),
                            'avSAR': np.average(np.average(1-juries[i]['juriestSAR'],axis=1)),
                            'avSTR': np.average(np.average(1-juries[i]['juriestSTR'],axis=1)),
                            'avRAN': np.average(np.average(1-juries[i]['juriestRAN'],axis=1)),
                            'diffAV': np.average(np.average(1-juries[i]['juriestSAR'],axis=1)) - np.average(np.average(1-juries[i]['juriestSTR'],axis=1)),
                            })

    df = pd.DataFrame(data)
    df = df.sort_values(by=['N'])
    print(df[['N','diffAV','diffFrac']])
    return df

#%% 
### construct dataframe with results

# load results
if __name__ == '__main__':

    # prepare a list with stats, first parameterization
    fname = gzip.open(outputdir+'logitnormal-signals.pickle.gz','rb')
    allresLogitN = pickle.load(fname)
    fname.close()
    
    datalogN = []
    for i in range(len(allresLogitN)):
        datalogN.append({'N': allresLogitN[i]['baseargs']['N'],
                            'fracSAR': np.average(np.sum(1-allresLogitN[i]['juriestSAR'],axis=1)>=1),
                            'fracSTR': np.average(np.sum(1-allresLogitN[i]['juriestSTR'],axis=1)>=1),
                            'fracRAN': np.average(np.sum(1-allresLogitN[i]['juriestRAN'],axis=1)>=1),
                            'diffFrac': np.average(np.sum(1-allresLogitN[i]['juriestSAR'],axis=1)>=1) - np.average(np.sum(1-allresLogitN[i]['juriestSTR'],axis=1)>=1),
                            'avSAR': np.average(np.average(1-allresLogitN[i]['juriestSAR'],axis=1)),
                            'avSTR': np.average(np.average(1-allresLogitN[i]['juriestSTR'],axis=1)),
                            'avRAN': np.average(np.average(1-allresLogitN[i]['juriestRAN'],axis=1)),
                            'diffAV': np.average(np.average(1-allresLogitN[i]['juriestSAR'],axis=1)) - np.average(np.average(1-allresLogitN[i]['juriestSTR'],axis=1)),
                            })

    df = pd.DataFrame(datalogN)
    df = df.sort_values(by=['N'])
    print(df[['N','diffAV','diffFrac']])
    allresLogitN2 = allresLogitN

    # second parameterization
    fname = gzip.open(outputdir+'logitnormal-2-signals.pickle.gz','rb')
    allresLogitN = pickle.load(fname)
    fname.close()
    
    # prepare a list with stats
    datalogN = []
    for i in range(len(allresLogitN)):
        datalogN.append({'N': allresLogitN[i]['baseargs']['N'],
                            'fracSAR': np.average(np.sum(1-allresLogitN[i]['juriestSAR'],axis=1)>=1),
                            'fracSTR': np.average(np.sum(1-allresLogitN[i]['juriestSTR'],axis=1)>=1),
                            'fracRAN': np.average(np.sum(1-allresLogitN[i]['juriestRAN'],axis=1)>=1),
                            'diffFrac': np.average(np.sum(1-allresLogitN[i]['juriestSAR'],axis=1)>=1) - np.average(np.sum(1-allresLogitN[i]['juriestSTR'],axis=1)>=1),
                            'avSAR': np.average(np.average(1-allresLogitN[i]['juriestSAR'],axis=1)),
                            'avSTR': np.average(np.average(1-allresLogitN[i]['juriestSTR'],axis=1)),
                            'avRAN': np.average(np.average(1-allresLogitN[i]['juriestRAN'],axis=1)),
                            'diffAV': np.average(np.average(1-allresLogitN[i]['juriestSAR'],axis=1)) - np.average(np.average(1-allresLogitN[i]['juriestSTR'],axis=1)),
                            })

    df2 = pd.DataFrame(datalogN)
    df2 = df2.sort_values(by=['N'])
    print(df2[['N','diffAV','diffFrac']])

# %%
if __name__ == '__main__':
    
    
    fig, (ax2, ax) = plt.subplots(1,2,figsize=(figsize,figsize/2+.5), sharey=True)  

    # prepare canvas remove right and top borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(-2,20)
    ax.set_yticks([0,5,10,15])
    ticks = np.log(np.array([.01,.1,1,5,10,20]))
    ax.set_xticks(ticks)
    ax.set_xticklabels(['0.01', '0.1', '1', '5', '10', '$\infty$'], fontsize=labelfont)
    ax.set_xlabel('Precision (Log scale)', fontsize=labelfont)
    ax.plot(np.log(df['N']),100*df['diffFrac'],color=SARcolor, label='Juries with at least 1 minority')
    ax.plot(np.log(df['N']),100* df['diffAV'],'--',color=STRcolor, label='Fraction of minority in juries')
    ax.scatter(np.log(20),100* df[df['N']==np.inf]['diffFrac'],color=SARcolor,s=5)
    ax.scatter(np.log(20),100* df[df['N']==np.inf]['diffAV'],color=STRcolor,s=5)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylim(-2,20)
    ax2.set_yticks([0,5,10,15])
    ti2c2ks = np.log(np.array([.01,.1,1,5,10,20]))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(['0.01', '0.1', '1', '5', '10', '$\infty$'], fontsize=labelfont)
    ax2.set_xlabel('Precision (Log scale)', fontsize = labelfont)
   
    ax2.plot(np.log(df2['N']),100*df2['diffFrac'],color=SARcolor, )
    ax2.plot(np.log(df2['N']),100* df2['diffAV'],'--',color=STRcolor, )
    ax2.scatter(np.log(20),100* df2[df2['N']==np.inf]['diffFrac'],color=SARcolor,s=5)
    ax2.scatter(np.log(20),100* df2[df2['N']==np.inf]['diffAV'],color=STRcolor,s=5)
 
    ax2.set_ylabel(r'\% Points')
    ax.set_title('(b) Moderate polarization', fontsize=11)
    ax2.set_title('(a) Extreme polarization', fontsize=11)

    #labellines.labelLines(plt.gca().get_lines(), align=False, xvals=(-1.2,-1),bbox={'alpha': 0}, fontsize=10, )# align=False)
    fig.legend(loc='upper left', title='Difference '+rep_label+'-'+str_label, bbox_to_anchor=(0.3, 0.86), fontsize=8, title_fontsize=9, framealpha=1 )
    fig.tight_layout()
    if savefig:
        fig.savefig(imagedir+'std-logitnorm.pdf')

#%%
# plot the densities of c

if __name__ == '__main__':

    jurinf = allresLogitN2[39]
    av1b = np.average(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==1])
    av0b = np.average(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==0])
    densRANinfb = gaussian_kde(jurinf['juriesxRAN'].flatten())
    dens0infb = gaussian_kde(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==0])
    dens1infb = gaussian_kde(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==1])
    
    jurinf = allresLogitN[39]
    av1 = np.average(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==1])
    av0 = np.average(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==0])
    densRANinf = gaussian_kde(jurinf['juriesxRAN'].flatten())
    dens0inf = gaussian_kde(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==0])
    dens1inf = gaussian_kde(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==1])

    fig, (ax2, ax)= plt.subplots(1,2,figsize=(figsize,figsize/2))  
    x = np.arange(0,1,.01)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.plot(x,densRANinfb(x), color=RANcolor, )
    ax.plot(x,dens0infb(x), '--', color='dimgrey',)
    ax.plot(x,dens1infb(x),'--', color='dimgrey', )
    ax.axvline(x=av1b, color='grey', linestyle=':',)
    ax.axvline(x=av0b, color='grey', linestyle=':',)
    ax.set_xticks([0,av1b,av0b,1])
    ax.set_xticklabels(['0', '$E(f_b)$', '$E(f_a)$', '1'], fontsize=8)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.plot(x,densRANinf(x), color=RANcolor, label='$f$')
    ax2.plot(x,dens0inf(x), '--', color='dimgrey', label='$f_a, f_b$')
    ax2.plot(x,dens1inf(x),'--', color='dimgrey', )
    ax2.axvline(x=av1, color='grey', linestyle=':')
    ax2.axvline(x=av0, color='grey', linestyle=':',)
    ax2.set_xticks([0,av1,av0,1])
    ax2.set_xticklabels(['0', '$E(f_b)$', '$E(f_a)$', '1'], fontsize=8)

    ax2.legend(fontsize=8)
 
    # remove y ticks and ticklabels
    ax.set_yticks([])
    ax2.set_yticks([])
    ax.set_title('(b) Moderate polarization', fontsize=11)
    ax2.set_title('(a) Extreme polarization', fontsize=11)

    fig.tight_layout()
    if savefig:
        fig.savefig(imagedir+'std-density-1.pdf', transparent=True)



#%%
if __name__ == '__main__':

    # load results
    # fname = gzip.open(outputdir+'beta-1-5-r90-signals.pickle.gz','rb')
    # allres90 = pickle.load(fname)
    # fname.close()
    fname = gzip.open(outputdir+'beta-1-5-r75-signals.pickle.gz','rb')
    allres75 = pickle.load(fname)
    fname.close()
    stats75 = computeStats(allres75)
    nsim75 = len(stats75)

    fname = gzip.open(outputdir+'beta-2-5-r75-signals.pickle.gz','rb')
    allSignals = pickle.load(fname)
    allres75b = allSignals[:11]+[allSignals[-1]]
    fname.close()
    stats75b = computeStats(allres75b)
    nsim75b = len(stats75b)

#%%
# plot the results
if __name__ == '__main__':
    nsim75 = len(stats75)

    fig, (ax2, ax) = plt.subplots(1,2,figsize=(figsize,figsize/2+.5), sharey=True)  

    # prepare canvas remove right and top borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    
    ax.plot(stats75['N'][:12], 100*stats75['diffAV'][:12], '--', color=STRcolor, label='Fraction of minority jurors')
    ax.plot(stats75['N'][:12], 100*stats75['diffFrac'][:12],  color=SARcolor, label='Juries with minorities')
    l1= ax.scatter(90,100*stats75['diffAV'][nsim75-1], color=STRcolor, marker='.')
    ax.scatter(90,100*stats75['diffFrac'][nsim75-1], color=SARcolor, marker='.')
    ax.set_ylim(-1,16)
    ax.set_yticks(np.arange(0,20,5))
    ax.set_xticks([0,20,40,60,80,90])
    ax.set_xticklabels([0,20,40,60,80,'$\infty$'])
    ax.set_xlabel('Precision (Log scale)', fontsize=xylabelfont)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax2.set_xlabel('Number of signals')
    ax2.set_ylabel('Difference REP-STR')
    ax2.plot(stats75b['N'], 100*stats75b['diffAV'], '--', color=STRcolor, )
    ax2.plot(stats75b['N'], 100* stats75b['diffFrac'],  color=SARcolor, )
    l12= ax2.scatter(80,100*stats75b['diffAV'][-1:], color=STRcolor, marker='.')
    ax2.scatter(80,100*stats75b['diffFrac'][-1:], color=SARcolor, marker='.')
    ax2.set_ylim(-1,16)
    ax2.set_yticks(np.arange(0,20,5))
    ax2.set_xticks([0,20,40,60,80,])
    ax2.set_xticklabels([0,20,40,60,'$\infty$'])

    ax2.set_ylabel('Percent point difference')
    ax.set_title('(b) Moderate polarization', fontsize=11)
    ax2.set_title('(a) Extreme polarization', fontsize=11)

    #labellines.labelLines(plt.gca().get_lines(), align=False, xvals=(-1.2,-1),bbox={'alpha': 0}, fontsize=10, )# align=False)
    fig.legend(loc='upper left', title='Difference REP-STR', bbox_to_anchor=(0.22, 0.87), fontsize=8, title_fontsize=9, framealpha=1 )
    fig.tight_layout()

    if savefig:
        fig.savefig(imagedir+'std-beta.pdf')

#%% plot the distributions
if __name__ == '__main__':

    x = np.arange(0,1,.01)
    jurinf = allres75[13]
    av1 = np.average(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==1])
    av0 = np.average(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==0])
    
    fig, (ax, ax2)= plt.subplots(1,2,figsize=(figsize,figsize/2))  
    x = np.arange(0,1,.01)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    densRANinf = gaussian_kde(jurinf['juriesxRAN'].flatten())
    dens0inf = gaussian_kde(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==0])
    dens1inf = gaussian_kde(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==1])

    ax.plot(x,densRANinf(x), color=RANcolor,   label='$f$')
    ax.plot(x,dens0inf(x), '--', color='dimgrey', label='$f_a, f_b$')
    ax.plot(x,dens1inf(x),'--', color='dimgrey',  )
    ax.axvline(x=av1, color='grey', linestyle=':', )
    ax.axvline(x=av0, color='grey', linestyle=':', )
    ax.set_xticks([0,av0,av1,1])
    ax.set_xticklabels(['0', '$E(f_b)$', '$E(f_a)$', '1'], fontsize=8)
    ax.set_yticks([])
    fig.tight_layout()

    x = np.arange(0,1,.01)
    jurinf = allres75b[-1]
    av1 = np.average(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==1])
    av0 = np.average(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==0])
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    x = np.arange(0,1,.01)
    densRANinf = gaussian_kde(jurinf['juriesxRAN'].flatten())
    dens0inf = gaussian_kde(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==0])
    dens1inf = gaussian_kde(jurinf['juriesxRAN'].flatten()[jurinf['juriestRAN'].flatten()==1])

    ax2.plot(x,densRANinf(x), color=RANcolor,   label='$f$')
    ax2.plot(x,dens0inf(x), '--', color='dimgrey', label='$f_a')
    ax2.plot(x,dens1inf(x),'--', color='dimgrey',  label='$f_b')
    ax2.axvline(x=av1b, color='grey', linestyle=':', )
    ax2.axvline(x=av0b, color='grey', linestyle=':', )
    ax2.set_xticks([0,av0b,av1b,1])
    ax2.set_xticklabels(['0', '$E(f_b)$', '$E(f_a)$', '1'], fontsize=8)
    ax2.set_yticks([])
    fig.tight_layout()
    ax2.set_title('(b) Moderate polarization', fontsize=11)
    ax.set_title('(a) Extreme polarization', fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if savefig:
        fig.savefig(imagedir+'std-density-beta.pdf', transparent=True)


# %%
