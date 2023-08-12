#!/usr/bin/env python3
"""
Created on Wed May 31 2023
@author: moroa
"""
#%%
import numpy as np
from class_model_types import Jurymodel
import  matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import pickle, gzip
import labellines 
# import statsmodels.api as sm
import sys
from copy import deepcopy
from multiprocessing import Pool,freeze_support
from scipy.stats import binom, norm
from scipy.special import beta as beta_func
from math import comb as choose


class Jury_statdisc(Jurymodel):
    '''
    Extend Jurymodel class to allow for statistical discrimination
    Arguments: N= number of signals
    '''
    def __init__(self, N=1, *args, **kwargs):
        self.N = N
        super().__init__(*args, **kwargs)
        self.initialize()

    def initialize(self):
        ''' initialize the model by computing distributions and expected values
        '''
        self.big_grid =  np.array([])
        self.pmf0, self.grid0 = self.pmfCompute(self.fx0)
        self.pmf1, self.grid1 = self.pmfCompute(self.fx1)
        self.pmf = self.R * self.pmf1 +  (1 - self.R) * self.pmf0 

        # merge grid0 and grid1 into a single grid of unique values        
        grid =  np.append(self.grid0,self.grid1)
        pmf = np.append(self.pmf0*(1-self.R),self.pmf1*self.R)

        # sort grid and pmf in unison according to grid
        a = dict.fromkeys(grid,0)
        for key,val in enumerate(grid):
            a[val] += pmf[key]
        self.big_grid = np.array(list(a.keys()))
        self.pmf = np.array(list(a.values()))
        self.mu = self.muCompute()
        
        self.computeV()

        return None

    def pmfCompute(self,fx):
        ''' compute the distribution of conditional expected conviction probabilities
        '''
        if fx['f']=='betaposterior':
            al = fx['al']
            bet = fx['bet']
            big_grid = np.array([])
            pmf = np.array([])
            if fx['f']=='betaposterior': 
                for success in range(self.N+1):
                    value = (al+success)/(al+bet+self.N)
                    freq = choose(self.N,success) * beta_func(success+al,self.N-success+bet)/beta_func(al,bet) 
                    big_grid = np.append(big_grid,value)
                    pmf = np.append(pmf,freq)
        
        elif fx['f']=='logit-normal':
            # the pmf of the conditional expectation after the noisy signal
            # is itself a logit-normal distribution, with variance parameter multiplied
            # by alpha = sig^2/(sig^2 + 1/N^2)
            
            # need to reset the big_grid, (this could be handled better)
            big_grid = np.arange(0+self.delta/2,1,self.delta)
            self.big_grid = big_grid
            alpha = fx['sig']**2 / (fx['sig']**2 + 1/self.N**2)
            condexpfx = deepcopy(fx)
            condexpfx['sig'] = fx['sig']*alpha
            pmf = super().pmfCompute(condexpfx)

        return pmf, big_grid
    
    def integrate(self,lb,ub):
        
        # I tend to forget that the integrand here is the CDF not the PDF!!!
        pdf = np.cumsum(self.pmf)
        rlb = np.where(self.big_grid >= lb)[0][0]
        rub = np.where(self.big_grid <= ub)[0][-1]
        area = pdf #(np.append(self.big_grid[1:],1)-self.big_grid)*pdf
        expval = sum(area[rlb:rub+1]) 

        return expval
    
    def fxdraw(self):

        '''
            draw jury pool expected conviction probabilities
        '''
        
        # draw type
        jtype = np.random.choice(2, self.poolSize, p=[1-self.R, self.R])
        x = np.zeros(len(jtype))

        if self.fx0['f']=='betaposterior':
            # draw true conviction probability
            if sum(jtype==0)>0:
                x[jtype==0] = np.random.beta(self.fx0['al'],self.fx0['bet'],sum(jtype==0))
            if sum(jtype==1)>0:
                x[jtype==1] = np.random.beta(self.fx1['al'],self.fx1['bet'],sum(jtype==1))

            # draw signals from binomial distribution
            t = np.random.binomial(self.N, x)

            # compute posterior from beta-binomial conjugacy
            x[jtype==0] = (t[jtype==0] + self.fx0['al'])/(self.N + self.fx0['al'] + self.fx0['bet'])
            x[jtype==1] = (t[jtype==1] + self.fx1['al'])/(self.N + self.fx1['al'] + self.fx1['bet'])
        
        elif self.fx0['f']=='logit-normal':
            if sum(jtype==0)>0:
                # draw true conviction probability
                draw = np.random.normal(self.fx0['mu'],self.fx0['sig'],sum(jtype==0))
                # draw a noisy signal
                signal = draw + np.random.normal(0,1/self.N,sum(jtype==0))
                # compute the expected value of the posterior
                alpha = self.fx0['sig']**2 / (self.fx0['sig']**2 + 1/self.N**2)
                cond_exp = self.fx0['mu']*(1-alpha) + signal * alpha 
                # actual draw from the posterior for the logit-normal
                x[jtype==0] = self.fx0['lb'] + (self.fx0['ub']-self.fx0['lb'])*np.exp(cond_exp)/(1+np.exp(cond_exp))
            if sum(jtype==1)>0:
                draw = np.random.normal(self.fx1['mu'],self.fx1['sig'],sum(jtype==1))
                signal = draw + np.random.normal(0,1/self.N,sum(jtype==1))
                alpha = self.fx1['sig']**2 / (self.fx1['sig']**2 + 1/self.N**2)
                cond_exp = self.fx1['mu']*(1-alpha) + signal * alpha
                x[jtype==1] = self.fx1['lb'] + (self.fx1['ub']-self.fx1['lb'])*np.exp(cond_exp)/(1+np.exp(cond_exp))
            
        return x, jtype

    def computeV(self):
        ''' recursively compute subgames value
        '''
        # reduce typing
        J = self.J
        D = self.D
        P = self.P

        # initialize matrices
        V = np.empty((J+1, D+1, P+1))*np.nan
        a = np.empty((J+1, D+1, P+1))*np.nan
        b = np.empty((J+1, D+1, P+1))*np.nan

        for d in range(0, D+1):
            for p in range(0, P+1):
                for j in range(0, J+1) :

                    # a: prosecution thresholds, b: defense thresholds
                    if p*j > 0:
                        a[j, d, p] = V[j, d, p-1] / V[j-1, d, p]
                        gridlessa = np.where(self.big_grid < a[j, d, p])[0]
                        if len(gridlessa) == 0:
                            athresh = -1
                        else:
                            athresh = gridlessa[-1]
                    if d*j > 0:
                        b[j, d, p] = V[j, d-1, p] / V[j-1, d, p]
                        gridmoreb = np.where(self.big_grid > b[j, d, p])[0]
                        if len(gridmoreb) == 0:
                            bthresh = len(self.big_grid)+1
                        else:
                            bthresh = gridmoreb[0]
                    
                    if d==0 and j*p != 0:

                        # only prosecution has challenges left
                        V[j, 0, p] = V[j, 0, p-1] * sum(self.pmf[:athresh+1]) + V[j-1, 0, p] * sum(self.pmf[athresh+1:]*self.big_grid[athresh+1:])

                    elif p==0 and j*d != 0:

                        # only defense has challenges left
                        V[j, d, 0] = V[j, d-1, 0] * sum(self.pmf[bthresh:]) + V[j-1, d, 0] * sum(self.pmf[:bthresh]*self.big_grid[:bthresh])

                    elif j*p*d != 0 :

                        # both have challenges left
                        V[j, d, p] = V[j, d, p-1]* sum(self.pmf[:athresh+1]) + V[j-1, d, p] * sum(self.pmf[athresh+1:bthresh]*self.big_grid[athresh+1:bthresh]) + V[j, d-1, p]* sum(self.pmf[bthresh:])
                    
                    else:

                        # Theorem 5 and Definition 4
                        V[j, d, p] = self.mu ** j

                    pass
                
        # save into object and return
        self.V = V
        self.a = a
        self.b = b

        return V,a,b   

def Jurypool(kwargs,show=''):
    njuries = 50000
    print('**', kwargs['N'], '**')
    mod = Jury_statdisc(**kwargs)
    jur = mod.manyJuries(njuries)
    jur['baseargs'] = kwargs
    return jur

#%% quick test
if __name__ == '__main__':

    baseargs = {'J' : 12,'D' : 6, 'P' : 6, 
            'R' : 0.25,
            'fx0' : {'f': 'betaposterior', 'al' : 1, 'bet': 5},
            'fx1' : {'f': 'betaposterior', 'al' : 5, 'bet': 1},
            # 'fx0' : {'f': 'uniform', 'lb' : 0, 'ub': .2},
            # 'fx1' : {'f': 'uniform', 'lb' : .2, 'ub': 1},
            'print_option': 1,
            'delta' : 1e-4,
            'seed' : 2443,
            }
    
    jur = Jury_statdisc(5,**baseargs)
    sim = jur.manyJuries(50000)

#%%
if __name__ == '__main__':

    baseargs = {'J' : 12,'D' : 6, 'P' : 6, 
            'R' : 0.25,
            'fx0' : {'f': 'beta', 'al' : 1, 'bet': 5},
            'fx1' : {'f': 'beta', 'al' : 5, 'bet': 1},
            # 'fx0' : {'f': 'uniform', 'lb' : 0, 'ub': .2},
            # 'fx1' : {'f': 'uniform', 'lb' : .2, 'ub': 1},
            'print_option': 1,
            'delta' : 1e-4,
            'seed' : 2443,
            }
   
    mod2 = Jurymodel(**baseargs)
    jur2 = mod2.manyJuries(50000)

    # combine juries simulations from three vectors  into a data frame
    df = pd.DataFrame({'RAN': sim['juriesxRAN'].flatten(), 'STR': sim['juriesxSTR'].flatten(), 'SAR': sim['juriesxSAR'].flatten()})
    df['RANbase'] = jur2['juriesxRAN'].flatten()
    df['STRbase'] = jur2['juriesxSTR'].flatten()
    df['SARbase'] = jur2['juriesxSAR'].flatten()


#%%
# if __name__ == '__main__':
#     str = sns.displot(df[['STRbase','STR']], kind="kde", )
#     sar = sns.displot(df[['SARbase','SAR']], kind="kde", )
#     ran = sns.displot(df[['RANbase','RAN']], kind="kde",)
#%%
if __name__ == '__main__':
    fig, ax1 = plt.subplots(1,1, figsize=(5,4))
    ax2 = ax1.twinx()
    from scipy.stats import gaussian_kde
    densSTR = gaussian_kde(df['STRbase'])
    densSAR = gaussian_kde(df['SARbase'])
    #ax1.hist(df['STR'], bins=np.unique(df['STR']), color='olive')
    x=np.arange(0,1,.01)
    ax1.hist(df['STR'], bins=np.unique(df['STR'])-np.unique(df['STR'])[0]/2, color='olive', alpha=.6)
    ax1.hist(df['SAR'], bins=np.unique(df['SAR'])-np.unique(df['SAR'])[0]/2, color='darkorange', alpha=.6, rwidth=.4)
    ax2.plot(x,densSTR(x), color='olive')
    ax2.plot(x,densSAR(x), color='darkorange')
# %%

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
    
    freeze_support()

    nSignals = [0,1,2,3,4,5, 7,10,13,16,20,24, 28,30,35,40,50, 60,70,80,90,100, ]

    # draw juries from statistical discrimination model 
    # with different number of signals
    iterlist = []
    for signals in nSignals:
        newkw = deepcopy(baseargs)
        newkw['N'] = signals
        iterlist.append((newkw))
        
    nprocs = 6

    pool = Pool(processes=nprocs)
    allSignals = pool.map(Jurypool,(iterlist))
    pool.close()
    pool.join()

    for idx,maj in enumerate(allSignals):
        maj['baseargs'] = iterlist[idx]

    #add simulations from with baseline model
    baseargs = {'J' : 12,'D' : 6, 'P' : 6, 
            'R' : 0.75,
            'fx0' : {'f': 'beta', 'al' : 1, 'bet': 5},
            'fx1' : {'f': 'beta', 'al' : 5, 'bet': 1},
            'print_option': 1,
            'delta' : 1e-4,
            'seed' : 2443,
            }
    newkw = deepcopy(baseargs)
    infsignals = Jurymodel(**baseargs).manyJuries(50000)
    newkw['N'] = np.inf
    infsignals['baseargs'] = newkw
    allSignals.append(infsignals)

    # save results
    fname = gzip.open('Output/beta-1-5-r75-signals.pickle.gz','wb')
    pickle.dump(allSignals,fname)
    fname.close()



#%%   
# the following is needed because multiprocess imports the __main__
# module for each process, so the entire file is executed multiple times
# see https://stackoverflow.com/questions/31858352/why-does-pool-run-the-entire-file-multiple-times
if __name__ == '__main__':

    baseargs = {'J' : 12,'D' : 6, 'P' : 6, 
            'R' : 0.90,
            'fx0' : {'f': 'betaposterior', 'al' : 1, 'bet': 5},
            'fx1' : {'f': 'betaposterior', 'al' : 5, 'bet': 1},
            'print_option': 1,
            'delta' : 1e-4,
            'seed' : 2443,
            }
    
    freeze_support()

    nSignals = [0,1,2,3,4,5, 7,10,13,16,20,24, 28,30,35,40,50, 60,70,80,90,100, ]

    # draw juries from statistical discrimination model 
    # with different number of signals
    iterlist = []
    for signals in nSignals:
        newkw = deepcopy(baseargs)
        newkw['N'] = signals
        iterlist.append((newkw))
        
    nprocs = 6

    pool = Pool(processes=nprocs)
    allSignals = pool.map(Jurypool,(iterlist))
    pool.close()
    pool.join()

    for idx,maj in enumerate(allSignals):
        maj['baseargs'] = iterlist[idx]

    # add simulations from with baseline model

    baseargs = {'J' : 12,'D' : 6, 'P' : 6, 
            'R' : 0.90,
            'fx0' : {'f': 'beta', 'al' : 1, 'bet': 5},
            'fx1' : {'f': 'beta', 'al' : 5, 'bet': 1},
            'print_option': 1,
            'delta' : 1e-4,
            'seed' : 2443,
            }
    newkw = deepcopy(baseargs)
    infsignals = Jurymodel(**baseargs).manyJuries(50000)
    newkw['N'] = np.inf
    infsignals['baseargs'] = newkw
    allSignals.append(infsignals)

    # save results
    fname = gzip.open('Output/beta-1-5-r90-signals.pickle.gz','wb')
    pickle.dump(allSignals,fname)
    fname.close()

#%%
if __name__ == '__main__':

    # load results
    fname = gzip.open('Output/beta-1-5-r90-signals.pickle.gz','rb')
    allres90 = pickle.load(fname)
    fname.close()
    fname = gzip.open('Output/beta-1-5-r75-signals.pickle.gz','rb')
    allres75 = pickle.load(fname)
    fname.close()
    
    # prepare a list with stats
    data90 = []
    for i in range(len(allres90)):
        data90.append({'nSignals': allres90[i]['baseargs']['N'],
                    'avSAR': np.average(np.average(1-allres90[i]['juriestSAR'],axis=1)),
                    'avSTR': np.average(np.average(1-allres90[i]['juriestSTR'],axis=1)),
                    'avRAN': np.average(np.average(1-allres90[i]['juriestRAN'],axis=1)),
                    'diffAV': np.average(np.average(1-allres90[i]['juriestSAR'],axis=1)) - np.average(np.average(1-allres90[i]['juriestSTR'],axis=1)),
                    'fracSAR': np.average(np.sum(1-allres90[i]['juriestSAR'],axis=1)>=1),
                    'fracSTR': np.average(np.sum(1-allres90[i]['juriestSTR'],axis=1)>=1),
                    'fracRAN': np.average(np.sum(1-allres90[i]['juriestRAN'],axis=1)>=1),
                    'diffFrac': np.average(np.sum(1-allres90[i]['juriestSAR'],axis=1)>=1) - np.average(np.sum(1-allres90[i]['juriestSTR'],axis=1)>=1),
                    })
    stats90 = pd.DataFrame(data90)
    data75 = []
    for i in range(len(allres75)):
        data75.append({'nSignals': allres75[i]['baseargs']['N'],
                    'avSAR': np.average(np.average(1-allres75[i]['juriestSAR'],axis=1)),
                    'avSTR': np.average(np.average(1-allres75[i]['juriestSTR'],axis=1)),
                    'avRAN': np.average(np.average(1-allres75[i]['juriestRAN'],axis=1)),
                    'diffAV': np.average(np.average(1-allres75[i]['juriestSAR'],axis=1)) - np.average(np.average(1-allres75[i]['juriestSTR'],axis=1)),
                    'fracSAR': np.average(np.sum(1-allres75[i]['juriestSAR'],axis=1)>=1),
                    'fracSTR': np.average(np.sum(1-allres75[i]['juriestSTR'],axis=1)>=1),
                    'fracRAN': np.average(np.sum(1-allres75[i]['juriestRAN'],axis=1)>=1),
                    'diffFrac': np.average(np.sum(1-allres75[i]['juriestSAR'],axis=1)>=1) - np.average(np.sum(1-allres75[i]['juriestSTR'],axis=1)>=1),
                    })
    
    # put everything in a data frame
    stats75 = pd.DataFrame(data75)
    #sort the data frame buy nSignals
    stats75 = stats75.sort_values(by=['nSignals'])

#%%
if __name__ == '__main__':
    # plot the results
    nsim90 = len(stats90)
    nsim75 = len(stats75)
    fix, ax= plt.subplots(1,1,figsize=(3.5,3.5))

    # prepare canvas remove right and top borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel('Number of signals')
    ax.set_ylabel('Difference')

    ax.plot(stats90['nSignals'], stats90['diffAV'], ':',color='olive', label='Fraction of minorities')
    ax.plot(stats90['nSignals'], stats90['diffFrac'],':',  color='darkorange', label='Juries with at least one minority')
    ax.scatter(107,stats90['diffAV'][nsim90-1], color='olive', marker='x')
    ax.scatter(107,stats90['diffFrac'][nsim90-1], color='darkorange', marker='x')
    ax.plot(stats75['nSignals'], stats75['diffAV'], color='olive', label='Fraction of minorities')
    ax.plot(stats75['nSignals'], stats75['diffFrac'],  color='darkorange', label='Juries with at least one minority')
    l1= ax.scatter(107,stats75['diffAV'][nsim75-1], color='olive', marker='.')
    ax.scatter(107,stats75['diffFrac'][nsim75-1], color='darkorange', marker='.')
    ax.set_ylim(-0.01,.15)
    #ax.set_yticks(np.arange(0,.2,.05))
    labellines.labelLines(plt.gca().get_lines(), xvals=[50,50], yoffsets=[.05,.15], align=False, bbox={'alpha': 0}, fontsize=10, )# align=False)
    # add tick at 55
    # ax.set_xticks([0,10,20,30,40,50,107])
    #ax.set_xticklabels([0,10,20,30,40,50,'$\infty$'])

#%%
if __name__ == '__main__':
    fig, ax= plt.subplots(1,1,figsize=(3.5,3.5))

    # prepare canvas remove right and top borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    nsim75 = len(stats75)
    ax.set_xlabel('Number of signals')
    ax.set_ylabel('Juries with at least one minority')
    ax.plot(stats75['nSignals'], stats75['avSAR'], ':', color='olive', label='Fraction of minorities')
    ax.plot(stats75['nSignals'], stats75['avSTR'], ':', color='darkorange', label='Juries with at least one minority')
    ax.scatter(107,stats75['avSAR'][nsim75-1], color='olive', marker='x')
    ax.scatter(107,stats75['avSTR'][nsim75-1], color='darkorange', marker='x')

if __name__ == '__main__':
    fig, ax= plt.subplots(1,1,figsize=(3.5,3.5))

    # prepare canvas remove right and top borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    nsim75 = len(stats75)
    ax.set_xlabel('Number of signals')
    ax.set_ylabel('Juries with at least one minority')
    ax.plot(stats75['nSignals'], stats75['avSAR'], ':', color='olive', label='Fraction of minorities')
    ax.plot(stats75['nSignals'], stats75['avSTR'], ':', color='darkorange', label='Juries with at least one minority')
    ax.scatter(107,stats75['avSAR'][nsim75-1], color='olive', marker='x')
    ax.scatter(107,stats75['avSTR'][nsim75-1], color='darkorange', marker='x')

    #%%
    # plot the all distributions of juriesxRAN in allrsults
    # first prepare the data
    df = pd.DataFrame(allres75[0]['juriesxRAN'].flatten(), columns=[allres90[0]['baseargs']['N']])
    for i in range(1,len(allres90)):
        df[allres90[i]['baseargs']['N']] = allres90[i]['juriesxRAN'].flatten()

#%%
if __name__ == '__main__':

    nsample = 6
    data = allSignals[nsample]
    RAN = data['juriesxRAN'].flatten()
    SAR = data['juriesxSAR'].flatten()
    STR = data['juriesxSTR'].flatten()
    unique = np.unique(STR)

    data = allSignals[len(allSignals)-1]
    RANdata = data['juriesxRAN'].flatten()
    SARdata = data['juriesxSAR'].flatten()
    STRdata = data['juriesxSTR'].flatten()

    fig, ax1 = plt.subplots(1,1, figsize=(5,4))
    ax2 = ax1.twinx()
    from scipy.stats import gaussian_kde
    densSTR = gaussian_kde(STRdata)
    densSAR = gaussian_kde(SARdata)
    x=np.arange(0,1,.01)
    ax1.hist(STR, bins=unique, color='olive', alpha=.6)
    ax1.hist(SAR, bins=unique, color='darkorange', alpha=.6, rwidth=.4)
    ax2.plot(x,densSTR(x), color='olive')
    ax2.plot(x,densSAR(x), color='darkorange')
    
# %%
