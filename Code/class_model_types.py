#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:15:52 2020

"""
#%%
import numpy as np
import sys,time
from scipy.stats import uniform, beta, norm

class Jurymodel():

    ''' defines Strike&Replace model

    Parameters:
        J : number of jurors
        D : number of defense challenges
        P : number of prosecution challenges
        R : proportion of type 1 jurors (note: = 1-r in the paper's notation)
        fx0: dict defining x distribution and its parameters for type 0 jurors
        fx1: dict defining x distribution and its parameters for type 1 jurors
            uniform example: {'f': 'uniform', 'lb' : 0, 'ub': 0.75},
            beta example: {'f': 'beta', 'al': 2, 'bet': 4}
        delta : precision of discrete approximations
    '''

    def __init__(self,
                 J=12,
                 D=6,
                 P=6,
                 R=0.5,
                 R2=0,
                 seed = 0,
                 fx0={'f': 'uniform', 'lb' : 0, 'ub': 0.75},
                 fx1={'f': 'uniform', 'lb' : 0.25, 'ub': 1},
                 fx2={'f': 'uniform', 'lb' : 0.25, 'ub': 1},
                 delta = 1e-4,
                 print_option = 1
        ):

        self.seed = seed
        self.J = J
        self.D = D
        self.P = P
        self.R = R
        self.R2 = R2
        self.fx0 = fx0
        self.fx1 = fx1
        self.fx2 = fx2
        self.delta = delta
        self.print_option = print_option

        self.V = np.zeros((J+1,D+1,P+1))*np.nan
        self.a = np.zeros((J+1,D+1,P+1))*np.nan
        self.b = np.zeros((J+1,D+1,P+1))*np.nan
        self.poolSize = self.J + self.P + self.D

        if self.seed != 0:
            np.random.seed(self.seed)

        # initialize the model if class is Jurymodel
        if self.__class__.__name__ == 'Jurymodel':
            self.initialize()

        # do some computations only once
    def initialize(self):
        ''' initialize the model by computing distributions and expected values
        '''

        self.big_grid =  np.arange(0+self.delta/2,1,self.delta)
        self.pmf0 = self.pmfCompute(self.fx0)
        self.pmf1 = self.pmfCompute(self.fx1)
        self.pmf2 = self.pmfCompute(self.fx2)
        self.pmf = self.R * self.pmf1 + self.R2 * self.pmf2 + (1 - self.R - self.R2) * self.pmf0
        self.mu = self.muCompute()
        
        self.computeV()

        return None

    def pmfCompute(self,fx):

        # Empirical distribution of x for type 0 depending on given distribution and its parameters
        if fx['f'] == 'uniform':

            dist = uniform(loc = fx['lb'], scale = fx['ub']-fx['lb'])
            pmf = dist.pdf(self.big_grid)*self.delta
            pmf = pmf/sum(pmf)
            # pmf[-1] = pmf[-1] - (sum(pmf) - 1)

            return pmf

        if fx['f'] == 'beta':

            #sanity check
            if fx['al']<=0 or fx['bet']<=0:
                sys.exit('beta with negative pars',fx)

            dist = beta(a = fx['al'], b = fx['bet'])
            pmf = dist.pdf(self.big_grid)*self.delta
            pmf = pmf/sum(pmf)
            # pmf[-1] = pmf[-1] - (sum(pmf) - 1)

            return pmf
    
        if fx['f'] == 'logit-normal':

            mu, sig, lb, ub = fx['mu'], fx['sig'], fx['lb'], fx['ub']    
            
            #sanity check
            if sig<=0:
                sys.exit('logitnormal with negative std',fx)
            
            dist = norm(mu, sig)
            mask = (self.big_grid >= lb) & (self.big_grid <= ub)
            logittrans = np.log((self.big_grid[mask]-lb)/(ub-self.big_grid[mask]))
            pmf = self.big_grid * 0
            pmf[mask] = dist.pdf(logittrans)*(ub-lb)/((ub-self.big_grid[mask])*(self.big_grid[mask]-lb))*self.delta
            pmf = pmf/sum(pmf)

            return pmf

    def muCompute(self):

        return sum(self.pmf*self.big_grid)

    def integrate(self,lb,ub):
        ''' computes the expected value of the distribution between lb and ub
        '''
        
        # I tend to forget that the integrand here is the CDF not the PDF!!!
        pdf = np.cumsum(self.pmf)
        rlb = np.where(self.big_grid >= lb)[0][0]
        rub = np.where(self.big_grid <= ub)[0][-1]
        expval = sum(pdf[rlb:rub+1]) * self.delta

        return expval

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

                    if p*j > 0:
                        a[j, d, p] = V[j, d, p-1] / V[j-1, d, p]
                    if d*j > 0:
                        b[j, d, p] = V[j, d-1, p] / V[j-1, d, p]

                    if d==0 and j*p != 0:

                        # Theorem 4
                        V[j, 0, p] = V[j-1, 0, p] * (1-self.integrate(a[j, d, p],1))

                    elif p==0 and j*d != 0:

                        # Theorem 3
                        V[j, d, 0] = V[j, d-1, 0] - V[j-1, d, 0] * self.integrate(0,b[j, d, p])

                    elif j*p*d != 0 :

                        # Theorem 2
                        V[j, d, p] = V[j, d-1, p] - V[j-1, d, p] * (self.integrate(a[j, d, p],b[j, d, p]))

                    else:

                        # Theorem 5 and Definition 4
                        V[j, d, p] = self.mu ** j

        # save into object and return
        self.V = V
        self.a = a
        self.b = b

        return V,a,b

    def fxdraw(self):

        '''
            draw jury pool from distribution
        '''
        
        # draw type
        jtype = np.random.choice(3, self.poolSize, p=[1-self.R-self.R2, self.R, self.R2])

        # draw conviction probability
        x = np.array([])
        
        # using internal beta function
        x = np.zeros(len(jtype))
        if sum(jtype==0)>0:
            if self.fx0['f'] == 'beta':
                x[jtype==0] = np.random.beta(self.fx0['al'],self.fx0['bet'],sum(jtype==0))
            if self.fx0['f'] == 'uniform':
                x[jtype==0] = np.random.uniform(self.fx0['lb'],self.fx0['ub'],sum(jtype==0))
            if self.fx0['f'] == 'logit-normal':
                draw = np.random.normal(self.fx0['mu'],self.fx0['sig'],sum(jtype==0))
                x[jtype==0] = self.fx0['lb'] + (self.fx0['ub']-self.fx0['lb'])*np.exp(draw)/(1+np.exp(draw))           
        if sum(jtype==1)>0:
            if self.fx1['f'] == 'beta':
                x[jtype==1] = np.random.beta(self.fx1['al'],self.fx1['bet'],sum(jtype==1))
            if self.fx1['f'] == 'uniform':
                x[jtype==1] = np.random.uniform(self.fx1['lb'],self.fx1['ub'],sum(jtype==1))
            if self.fx1['f'] == 'logit-normal':
                draw = np.random.normal(self.fx1['mu'],self.fx1['sig'],sum(jtype==1))
                x[jtype==1] = self.fx1['lb'] + (self.fx1['ub']-self.fx1['lb'])*np.exp(draw)/(1+np.exp(draw))
        if sum(jtype==2)>0:
            if self.fx2['f'] == 'beta':
                x[jtype==2] = np.random.beta(self.fx2['al'],self.fx2['bet'],sum(jtype==2))
            if self.fx2['f'] == 'uniform':
                x[jtype==2] = np.random.uniform(self.fx2['lb'],self.fx2['ub'],sum(jtype==2))
            if self.fx2['f'] == 'logit-normal':
                draw = np.random.normal(self.fx2['mu'],self.fx2['sig'],sum(jtype==2))
                x[jtype==2] = self.fx2['lb'] + (self.fx2['ub']-self.fx2['lb'])*np.exp(draw)/(1+np.exp(draw))
         
        return x, jtype

    def simulateSTRjury(self,draw={}):
 
        if draw is self.simulateSTRjury.__defaults__[0]:
            draw = self.fxdraw()
            
        x_vec = draw[0]
        t_vec = draw[1]

        juryID = x_vec.argsort()[self.P : self.P + self.J]
        chDID =  x_vec.argsort()[self.P + self.J : self.poolSize]
        chPID =  x_vec.argsort()[0 : self.P]

        jury = x_vec[juryID]
        juryt =  t_vec[juryID]

        # do we need these? keep for later
        chD = x_vec[chDID]
        chDt =  t_vec[chDID]
        chP = x_vec[chPID]
        chPt =  t_vec[chPID]
        
        return jury,juryt,chD,chDt,chP,chPt

    def simulateSARjury(self,draw={}):
 
        if draw is self.simulateSARjury.__defaults__[0]:
            draw = self.fxdraw()
            
        x_vec = draw[0]
        t_vec = draw[1]
        
        # array of accepted jurors, challgenged from d, and challenged from p
        jury = np.empty(self.J)
        juryt = np.empty(self.J,dtype=int)

        # arrays of challenges from Defense and Prosecution (c and t[ype])
        # reverting to append for challenges because may not use all challenges
        # pre-allocating returns weird values and hides number of challenges used
        chD = np.array([])
        chDt = np.array([])
        chP = np.array([])
        chPt = np.array([])

        j = self.J
        d = self.D
        p = self.P

        counter = 0

        while j>=1:

            # pick "counter" juror
            x = x_vec[counter]
            t = t_vec[counter]

            if d>=1: #d and p are the challenge numbers
                if x > self.b[j, d, p]:

                    # challenged from defense
                    chD = np.append(chD,x)
                    chDt = np.append(chDt, t)
                    d = d-1
                    counter = counter + 1
                    continue

            if p>=1:
                if x < self.a[j, d, p]:

                    # challenged from prosecution
                    chP = np.append(chP,x)
                    chPt = np.append(chPt, t)
                    p = p-1
                    counter = counter + 1
                    continue

            #accepted
            jury[j-1] = x
            juryt[j-1] = t
            j = j-1
            counter = counter + 1
        
        return jury,juryt,chD,chDt,chP,chPt
     
    def simulateRANjury(self,draw={}):

        if draw is self.simulateSTRjury.__defaults__[0]:
            draw = self.fxdraw()

        x_vec = draw[0]
        t_vec = draw[1]
        randomjury = np.random.choice(self.poolSize, self.J, replace=False)

        # select value and type with those indexes
        jury = x_vec[randomjury]
        juryt = t_vec[randomjury]

        # do we need these? keep for later
        chD = np.delete(x_vec,randomjury)
        chDt =  np.delete(t_vec,randomjury)
        chP = np.array([])
        chPt = np.array([])

        return jury,juryt,chD,chDt,chP,chPt

    def simulateJuryProc(self,procedure,draw={}):

        if draw is self.simulateJuryProc.__defaults__[0]:
            draw = self.fxdraw()
        if procedure == 'STR' :
            return self.simulateSTRjury(draw)
        if procedure == 'SAR':
            return self.simulateSARjury(draw)
        if procedure == 'RAN':
            return self.simulateRANjury(draw)

    def simulateJury(self, draw={}):

        # Draw jury panel
        if draw == self.simulateSTRjury.__defaults__[0]:
            draw = self.fxdraw()

        jurySAR, jurytSAR, chDSAR, chDtSAR ,chPSAR, chPtSAR = self.simulateJuryProc('SAR', draw)
        jurySTR, jurytSTR, chDSTR, chDtSTR, chPSTR, chPtSTR = self.simulateJuryProc('STR', draw)
        juryRAN, jurytRAN, chDRAN, chDtRAN, chPRAN, chPtRAN = self.simulateJuryProc('RAN', draw)

        #count challenges of [blacks, whites] by parties
        numch = (sum(chDtSTR),sum(chPtSTR))
        numChDSTR = np.array([chDtSTR.size-numch[0],numch[0]])
        numChPSTR = np.array([chPtSTR.size-numch[1],numch[1]])
        numch = (sum(chDtSAR),sum(chPtSAR))
        numChDSAR = np.array([chDtSAR.size-numch[0],numch[0]])
        numChPSAR = np.array([chPtSAR.size-numch[1],numch[1]])
        numTypesPool = (self.poolSize-sum(draw[1]),sum(draw[1]))
        numJury = sum(jurytRAN)
        numChRAN = np.array([numTypesPool[0]-self.J+numJury,numTypesPool[1]-numJury])

        return {'jurySAR': jurySAR, 'jurytSAR': jurytSAR,
                'jurySTR': jurySTR, 'jurytSTR': jurytSTR,
                'juryRAN': juryRAN, 'jurytRAN': jurytRAN,
                'juryPool': draw[0],  'jurytPool': draw[1],
                'numChDSTR': numChDSTR, 'numChPSTR': numChPSTR,
                'numChDSAR': numChDSAR, 'numChPSAR': numChPSAR,
                'numChRAN': numChRAN,
                }

    def manyJuries(self,N):

        # returns many juries
        juriestSAR = np.zeros((N,self.J))-9
        juriestSTR = np.zeros((N,self.J))-9
        juriestRAN = np.zeros((N,self.J))-9
        juriesxSAR = np.zeros((N,self.J))-9
        juriesxSTR = np.zeros((N,self.J))-9
        juriesxRAN = np.zeros((N,self.J))-9
        numChDSTR  = np.zeros((N,2))-9
        numChPSTR  = np.zeros((N,2))-9

        for jury in range(N):
            if self.print_option>=1:
                if jury % 5000 == 0:
                    print(jury)
            juries = self.simulateJury()
            juriestSAR[jury,:] = juries['jurytSAR']
            juriestSTR[jury,:] = juries['jurytSTR']
            juriestRAN[jury,:] = juries['jurytRAN']
            juriesxSAR[jury,:] = juries['jurySAR']
            juriesxSTR[jury,:] = juries['jurySTR']
            juriesxRAN[jury,:] = juries['juryRAN']
            numChDSTR[jury,:] = juries['numChDSTR']
            numChPSTR[jury,:] = juries['numChPSTR']

        pguiltSAR = np.prod(juriesxSAR,axis=1)
        pguiltSTR = np.prod(juriesxSTR,axis=1)
        pguiltRAN = np.prod(juriesxRAN,axis=1)

        return {'juriesxSAR': juriesxSAR, 'juriestSAR': juriestSAR,
                'juriesxSTR': juriesxSTR, 'juriestSTR': juriestSTR,
                'juriesxRAN': juriesxRAN, 'juriestRAN': juriestRAN,
                'numChDSTR': numChDSTR, 'numChPSTR': numChPSTR,
                'pguiltSAR': pguiltSAR, 'pguiltSTR': pguiltSTR, 'pguiltRAN': pguiltRAN,
                }

    def computeMoments(self,juriest,numChD,numChP):

        frT1inJuries = np.zeros(self.J)
        for idx,n in enumerate(range(self.J)):
            frT1inJuries[idx] = np.average(np.sum(juriest,axis=1)==idx)


        nT0inPool = (numChD+numChP)[:,0] + np.sum(1-juriest, axis=1)
        nT1inPool = (numChD+numChP)[:,1] + np.sum(juriest, axis=1)
        cutoffsT0 = np.percentile(nT0inPool,(0,33,66,100))
        cutoffsT1 = np.percentile(nT1inPool,(0,33,66,100))

        frT0chDall = numChD[:,0]/nT0inPool
        frT1chDall = numChD[:,1]/nT1inPool
        frT0chPall = numChP[:,0]/nT0inPool
        frT1chPall = numChP[:,1]/nT1inPool

        frT0chP = np.zeros(cutoffsT0.size-1)
        frT0chD = np.zeros(cutoffsT0.size-1)
        frT1chP = np.zeros(cutoffsT0.size-1)
        frT1chD = np.zeros(cutoffsT0.size-1)

        #adjustment to make sure it counts all juries
        cutoffsT0[0] += -1
        cutoffsT1[0] += -1
        for idx,n in enumerate(range(cutoffsT0.size-1)):
            mask = (nT0inPool>cutoffsT0[idx]) & (nT0inPool<=cutoffsT0[idx+1])
            frT0chD[idx] = np.average(frT0chDall[mask])
            frT0chP[idx] = np.average(frT0chPall[mask])
            mask = (nT1inPool>cutoffsT1[idx]) & (nT1inPool<=cutoffsT1[idx+1])
            frT1chD[idx] = np.average(frT1chDall[mask])
            frT1chP[idx] = np.average(frT1chPall[mask])

        return(frT1inJuries)


def Jurypool(kwargs):
    njuries = 20000
    print('**', kwargs['D'], '**')
    mod = Jurymodel(**kwargs)
    jur = mod.manyJuries(njuries)
    jur['baseargs'] = kwargs
    return jur

def Jurypool5(kwargs):
    njuries = 50000
    print('**', kwargs['R'], '**')
    mod = Jurymodel(**kwargs)
    jur = mod.manyJuries(njuries)
    jur['baseargs'] = kwargs
    return jur

def Jurypool10(kwargs):
    njuries = 100000
    print('**', kwargs['R'], '**')
    mod = Jurymodel(**kwargs)
    jur = mod.manyJuries(njuries)
    jur['baseargs'] = kwargs
    return jur


#%% test code
if __name__ == '__main__':

    a0, b0 = [1,5]
    a1, b1 = [5,1]
    R = .6594705
    baseargs = {'J' : 12,'D' : 6, 'P' : 6,
            'R' : R,
            'fx0' : {'f': 'beta', 'al' : a0, 'bet': b0},
            'fx1' : {'f': 'beta', 'al' : a1, 'bet': b1},
            # 'fx0' : {'f': 'uniform', 'lb' : a0, 'ub': b0},
            # 'fx1' : {'f': 'uniform', 'lb' : a1, 'ub': b1},
            'print_option': 1,
            'delta' : 1e-5,
            'seed' : 2443,
            }
    timenow = time.time()
    model = Jurymodel(**baseargs)

    j1 = model.fxdraw()
    s1 = model.simulateSTRjury(draw=j1)

    resultbeta = model.manyJuries(2501)
    print(np.average(resultbeta['juriesxRAN']))
    print(np.average(resultbeta['juriestRAN']))
    print('time: ',time.time()-timenow)

# %%
