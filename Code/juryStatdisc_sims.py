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

nprocs = 6

STRcolor = 'olive'
SARcolor = 'Darkorange'
RANcolor = 'SaddleBrown'

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

if __name__ == '__main__':
    mods = {}
    jurs = {}

    listprec = list(np.arange(0.01,0.1,.03))
    listprec += list(np.arange(.1,.3,.05))
    listprec += list(np.arange(.3,1.2,.05))
    listprec += list(np.arange(1.2,3,.5))
    listprec += list(np.arange(3,5,.5)) 
    listprec += [5,6,7,8,10]
    
    iterlist = []
    for N in listprec:
        newkw = deepcopy(baseargs)
        newkw['N'] = N
        iterlist.append((newkw))

    pool = Pool(processes=nprocs)
    allSignals = pool.map(Jurypool,(iterlist))
    pool.close()
    pool.join()

    for idx,maj in enumerate(allSignals):
        maj['baseargs'] = iterlist[idx]

    #add simulations from with baseline model
    newkw = deepcopy(baseargs)
    infsignals = Jurymodel(**newkw).manyJuries(njuries)
    newkw['N'] = np.inf
    infsignals['baseargs'] = newkw
    allSignals.append(infsignals)

    # save results
    fname = gzip.open(outputdir+'logitnormal-signals.pickle.gz','wb')
    pickle.dump(allSignals,fname)
    fname.close()

#%% second logitnormal model
baseargs2 = {'J' : 12,'D' : 6, 'P' : 6, 
        'R' : 0.75,
        'fx0' : {'f': 'logit-normal', 'mu' : -2, 'sig': 1, 'lb': 0, 'ub': 1},
        'fx1' : {'f': 'logit-normal', 'mu' : 2,  'sig': 1, 'lb': 0, 'ub': 1},
        'print_option': 1,
        'delta' : 1e-4,
        'seed' : 2443,
        }

if __name__ == '__main__':
    mods = {}
    jurs = {}

    listprec = list(np.arange(0.01,0.1,.03))
    listprec += list(np.arange(.1,.3,.05))
    listprec += list(np.arange(.3,1.2,.05))
    listprec += list(np.arange(1.2,3,.5))
    listprec += list(np.arange(3,5,.5)) 
    listprec += [5,6,7,8,10]
    
    iterlist = []
    for N in listprec:
        newkw = deepcopy(baseargs2)
        newkw['N'] = N
        iterlist.append((newkw))

    pool = Pool(processes=nprocs)
    allSignals = pool.map(Jurypool,(iterlist))
    pool.close()
    pool.join()

    for idx,maj in enumerate(allSignals):
        maj['baseargs'] = iterlist[idx]

    #add simulations from with baseline model
    newkw = deepcopy(baseargs2)
    infsignals = Jurymodel(**newkw).manyJuries(njuries)
    newkw['N'] = np.inf
    infsignals['baseargs'] = newkw
    allSignals.append(infsignals)

    # save results
    fname = gzip.open(outputdir+'logitnormal-2-signals.pickle.gz','wb')
    pickle.dump(allSignals,fname)
    fname.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#### results with beta-binomial


#%%   
# the following is needed because multiprocess imports the __main__
# module for each process, so the entire file is executed multiple times
# see https://stackoverflow.com/questions/31858352/why-does-pool-run-the-entire-file-multiple-times
if __name__ == '__main__':

    baseargs = {'J' : 12,'D' : 6, 'P' : 6, 
            'R' : 0.75,
            'fx0' : {'f': 'betaposterior', 'al' : 1, 'bet': 5},
            'fx1' : {'f': 'betaposterior', 'al' : 5, 'bet': 1},
            'print_option': 1,
            'delta' : 1e-4,
            'seed' : 2443,
            }

    nSignals = [0,2,4,8,12,16,20,25,35,50,70,80,90,100, ]
    
    # draw juries from statistical discrimination model 
    # with different number of signals
    iterlist = []
    for signals in nSignals:
        newkw = deepcopy(baseargs)
        newkw['N'] = signals
        iterlist.append((newkw))

    pool = Pool(processes=nprocs)
    allSignals = pool.map(Jurypool,(iterlist))
    pool.close()
    pool.join()

    for idx,maj in enumerate(allSignals):
        maj['baseargs'] = iterlist[idx]

    #add simulations from with baseline model
    newkw = deepcopy(baseargs)
    newkw['fx0']['f'] = 'beta'
    newkw['fx1']['f'] = 'beta'
    infsignals = Jurymodel(**newkw).manyJuries(njuries)
    newkw['N'] = np.inf
    infsignals['baseargs'] = newkw
    allSignals.append(infsignals)

    # save results
    fname = gzip.open(outputdir+'beta-1-5-r75-signals.pickle.gz','wb')
    pickle.dump(allSignals,fname)
    fname.close()

#%%
# second example of beta distribution, less polarized

# if __name__ == '__main__':

    baseargs = {'J' : 12,'D' : 6, 'P' : 6, 
            'R' : 0.75,
            'fx0' : {'f': 'betaposterior', 'al' : 2, 'bet': 5},
            'fx1' : {'f': 'betaposterior', 'al' : 5, 'bet': 2},
            'print_option': 1,
            'delta' : 1e-4,
            'seed' : 2443,
            }

    nSignals = [0,2,4,8,12,16,20,25,35,50,70,80,90,100, ]

    # draw juries from statistical discrimination model 
    # with different number of signals
    iterlist = []
    for signals in nSignals:
        newkw = deepcopy(baseargs)
        newkw['N'] = signals
        iterlist.append((newkw))

    pool = Pool(processes=nprocs)
    allSignals = pool.map(Jurypool,(iterlist))
    pool.close()
    pool.join()

    for idx,maj in enumerate(allSignals):
        maj['baseargs'] = iterlist[idx]

    # add simulations from with baseline model
    newkw = deepcopy(baseargs)
    newkw['fx0']['f'] = 'beta'
    newkw['fx1']['f'] = 'beta'
    infsignals = Jurymodel(**newkw).manyJuries(njuries)
    newkw['N'] = np.inf
    infsignals['baseargs'] = newkw
    allSignals.append(infsignals)

    # save results
    fname = gzip.open(outputdir+'beta-2-5-r75-signals.pickle.gz','wb')
    pickle.dump(allSignals,fname)
    fname.close()

