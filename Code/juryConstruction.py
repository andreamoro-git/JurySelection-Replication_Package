#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# In this file we construct juries and save data to be used elsewhere to
# draw figures and compute stats
#
# (note: let's try to keep only those actually used in the paper)
#
import gzip,pickle
from copy import deepcopy
import numpy as np
from multiprocessing import Pool,freeze_support
from class_model_types import Jurymodel

outputdir = '../Simulations/'
defaultNjuries = 50000
nprocs = 6

def Jurypool(kwargs,show=''):
    njuries = defaultNjuries
    print('**', kwargs['R'], '**')
    mod = Jurymodel(**kwargs)
    jur = mod.manyJuries(njuries)
    jur['baseargs'] = kwargs
    return jur

def JurypoolD(kwargs):
    njuries = defaultNjuries
    print('**', kwargs['D'], '**')
    mod = Jurymodel(**kwargs)
    jur = mod.manyJuries(njuries)
    jur['baseargs'] = kwargs
    return jur


#%%   
# the following is needed because multiprocess imports the __main__
# module for each process, so the entire file is executed multiple times
# see https://stackoverflow.com/questions/31858352/why-does-pool-run-the-entire-file-multiple-times
if __name__ == '__main__':
    freeze_support()

#%% 
# Beta, extreme polarization (for fig:atleast1-3beta)

    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.75,
                'fx0' : {'f': 'beta', 'al' : 1, 'bet': 5},
                'fx1' : {'f': 'beta', 'al' : 5, 'bet': 1},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-1-5-12j-75pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()


#%% 
# Beta, moderate polarization (for fig:atleast1-3beta)

    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.75,
                'fx0' : {'f': 'beta', 'al' : 2, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 2},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-2-4-12j-75pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()
    
    #%% Beta, mild polarization (for fig:atleast1-3beta)
    
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.75,
                'fx0' : {'f': 'beta', 'al' : 3, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 3},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-3-4-12j-75pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()

#%% Uniform, 1j, polarized (For fig:prop2-uni, Prop. 2)

    njuries = defaultNjuries
    baseargs = {'J' : 1,'D' : 1, 'P' : 1,
                'R' : 0.25,
                'fx0' : {'f': 'uniform', 'lb' : 0, 'ub': 0.1},
                'fx1' : {'f': 'uniform', 'lb' : 0.9, 'ub': 1},
                'print_option': 2,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultuni = model.manyJuries(njuries)
    resultuni['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'uni-0.1-0.9-1j-25pcT1.pickle.gz','wb')
    pickle.dump(resultuni,fname)
    fname.close()

#%%  Minority representation in size-1 juries (for figure fig:counter_b)

    baseargs = {'J' : 1,'D' : 1, 'P' : 1,
                'R' : 0.5,
                'fx0' : {'f': 'uniform', 'lb' : 0, 'ub': .5},
                'fx1' : {'f': 'uniform', 'lb' : .5, 'ub': 1},
                'print_option': 0,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    iterlist = []
    r = np.arange(0.02,0.41,0.02)
    
    for gsize in r:
            newkw = deepcopy(baseargs)
            newkw['R'] = 1-gsize
            newkw['fx0'] = {'f': 'uniform', 'lb' : 0, 'ub': gsize}
            newkw['fx1'] = {'f': 'uniform', 'lb' : gsize, 'ub': 1}
            iterlist.append((newkw))

    pool = Pool(processes=nprocs)
    allmajSizes = pool.map(Jurypool,(iterlist))
    pool.close()
    pool.join()

    for idx,maj in enumerate(allmajSizes):
        maj['baseargs'] = iterlist[idx]

    fname = gzip.open(outputdir + 'uni-1-1-1-MANYr.gz','wb')
    pickle.dump(allmajSizes,fname)
    fname.close()

#%% Representation of minority jurors ( for tab:betas-grouprep - Table 1)

    # Extreme, 1-r = .5
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.5,
                'fx0' : {'f': 'beta', 'al' : 1, 'bet': 5},
                'fx1' : {'f': 'beta', 'al' : 5, 'bet': 1},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-1-5-12j-50pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()
    
    # Extreme, 1-r = .90
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.90,
                'fx0' : {'f': 'beta', 'al' : 1, 'bet': 5},
                'fx1' : {'f': 'beta', 'al' : 5, 'bet': 1},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-1-5-12j-90pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()
    
    # Moderate, 1-r = .5
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.5,
                'fx0' : {'f': 'beta', 'al' : 2, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 2},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    
    fname = gzip.open(outputdir + 'beta-2-4-12j-50pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()
    
    # Moderate, 1-r = 0.9
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.90,
                'fx0' : {'f': 'beta', 'al' : 2, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 2},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    
    fname = gzip.open(outputdir + 'beta-2-4-12j-90pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()
    
    # Mild, 1-4 = -.75
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.5,
                'fx0' : {'f': 'beta', 'al' : 3, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 3},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-3-4-12j-50pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()
    
    # Mild, 1-r= .90
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.90,
                'fx0' : {'f': 'beta', 'al' : 3, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 3},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-3-4-12j-90pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()

#%% Number of challenges, for Figure fig:nchallenges

    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.8,
                'fx0' : {'f': 'beta', 'al' : 2, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 2},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    
    iterlist = []
    challenges = np.arange(1,21)
    
    # generate list of parameter sets to send to multiprocess
    for nchg in challenges:
            newkw = deepcopy(baseargs)
            newkw['D'] = nchg
            newkw['P'] = nchg
            iterlist.append((newkw))
        
    if __name__ == '__main__':
        freeze_support()
        pool = Pool(processes=nprocs)
        allchg = pool.map(JurypoolD,iterlist)
        pool.close()
        pool.join()
    
        # save parameter values togeter with data
        for idx,maj in enumerate(allchg):
            maj['baseargs'] = iterlist[idx]
    
        fname = gzip.open(outputdir + 'manyChallenges.pickle.gz','wb')
        pickle.dump(allchg,fname)
        fname.close()

#%% Representation in balanced groups
# TABLE 2, Slightly Asymmetric distributions

# Beta 1-5, 5-2, T1 = 50%
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.5,
                'fx0' : {'f': 'beta', 'al' : 1, 'bet': 5},
                'fx1' : {'f': 'beta', 'al' : 5, 'bet': 2},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-1-5-5-2-12j-50pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()

#%% TABLE 2, Slightly Asymmetric distributions
# Beta 2-4, 4-3, T1 = 50%
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.5,
                'fx0' : {'f': 'beta', 'al' : 2, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 3},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-2-4-4-3-12j-50pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()

#%% TABLE 2, Slightly Asymmetric distributions
# Beta 2-4, 4-3, T1 = 50%

    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.5,
                'fx0' : {'f': 'beta', 'al' : 2, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 3},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-2-4-4-3-12j-50pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()

#%% TABLE 2, Slightly Asymmetric distributions
# Beta 3-4, 4-4, T1 = 50%
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.5,
                'fx0' : {'f': 'beta', 'al' : 3, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 4},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-3-4-4-4-12j-50pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()

#%% TABLE 2, Slightly Asymmetric groups (extreme)
# Beta 1-5, 5-1, T1 = 55%
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.55,
                'fx0' : {'f': 'beta', 'al' : 1, 'bet': 5},
                'fx1' : {'f': 'beta', 'al' : 5, 'bet': 1},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-1-5-12j-55pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()

#%% TABLE 2, Slightly Asymmetric groups (moderate)
    # Beta 2-4, 4-2, T1 = 55%
    
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.55,
                'fx0' : {'f': 'beta', 'al' : 2, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 2},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-2-4-12j-55pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()

#%% TABLE 2, Slightly Asymmetric groups (mild)
# Beta 3-4, 4-3, T1 = 55%

    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.55,
                'fx0' : {'f': 'beta', 'al' : 3, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 3},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-3-4-12j-55pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()

#%% Appendix, uniform (for fig:prop1-uni)

    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.5,
                'fx0' : {'f': 'uniform', 'lb' : 0, 'ub': 1},
                'fx1' : {'f': 'uniform', 'lb' : 0, 'ub': 1},
                'print_option': 2,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultuni = model.manyJuries(njuries)
    resultuni['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'uni-12j-6-6.pickle.gz','wb')
    pickle.dump(resultuni,fname)
    fname.close()

#%% Appendix Table tab-app:betas-grouprep,
# minority representation when minority favor conviction

#note: we are drawing here the same distributions as in the main
#with T1 being now the minority, therefore statistics are computed
#averaging T1 jurors

#extreme T1=.1
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.1,
                'fx0' : {'f': 'beta', 'al' : 1, 'bet': 5},
                'fx1' : {'f': 'beta', 'al' : 5, 'bet': 1},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-1-5-12j-10pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()

#extreme, T1=.25
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.25,
                'fx0' : {'f': 'beta', 'al' : 1, 'bet': 5},
                'fx1' : {'f': 'beta', 'al' : 5, 'bet': 1},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-1-5-12j-25pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()
    
    #moderate, T1=.1
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.10,
                'fx0' : {'f': 'beta', 'al' : 2, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 2},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    
    fname = gzip.open(outputdir + 'beta-2-4-12j-10pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()
    
    #moderate, T1=.25
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.25,
                'fx0' : {'f': 'beta', 'al' : 2, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 2},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    
    fname = gzip.open(outputdir + 'beta-2-4-12j-25pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()
    
    #mild, T1=.1
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.1,
                'fx0' : {'f': 'beta', 'al' : 3, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 3},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-3-4-12j-10pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()
    
    #mild, T1=.25
    njuries = defaultNjuries
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
                'R' : 0.25,
                'fx0' : {'f': 'beta', 'al' : 3, 'bet': 4},
                'fx1' : {'f': 'beta', 'al' : 4, 'bet': 3},
                'print_option': 1,
                'delta' : 1e-4,
                'seed' : 2443,
                }
    
    model = Jurymodel(**baseargs)
    resultbeta = model.manyJuries(njuries)
    resultbeta['baseargs'] = baseargs
    
    fname = gzip.open(outputdir + 'beta-3-4-12j-25pcT1.pickle.gz','wb')
    pickle.dump(resultbeta,fname)
    fname.close()
