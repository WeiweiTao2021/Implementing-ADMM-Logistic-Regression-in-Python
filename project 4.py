#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:10:28 2021

@author: weiweitao
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import glob

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))
    
def loss(y, p_hat):
    loss = -np.mean(y*(np.log(p_hat)) + (1-y)*np.log(1-p_hat))
    return loss


def target_beta(beta,*args):
    u = args[0]
    # global ground truth
    b = args[1]
    # rho 
    rho = args[2]
    
    y = args[3]
    x = args[4]
    
    p_hat = sigmoid(np.dot(x, beta))
    
    L0 = loss(y, p_hat) 
    L = L0 + (rho/2.0) * np.sum((beta - b + u)**2)
    #print(L0, L)
    #return(loss(y, p_hat))
    return(L)


################Baseline Model
path = r'/Users/weiweitao/Desktop/Stony Brook/Fall 2021 Courses/AMS 598/Project 4/' # use your path
all_files = glob.glob(path + "/project*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

y = frame['y']
X = frame[set(df.columns) - set('y')]
## standardize all features
scaler = StandardScaler().fit(X)
X_std = scaler.transform(X)
X_std = np.concatenate((np.ones((X_std.shape[0], 1)), X_std), axis=1)

log_reg = sm.Logit(y, X_std).fit()
print(log_reg.params)

pd.DataFrame(log_reg.params).to_csv('reference.csv')



############ initialize values
# initialize beta matrix: 10*p

rhos = [0.1]
for rho in rhos:
    
    p = 26
    beta = np.zeros(10*p).reshape(10,p)
    beta_new = np.zeros(10*p).reshape(10,p)
    b = np.mean(beta, axis = 0)
    u = np.zeros(10*p).reshape(10,p)
    k = 0
    
    li = []

    while(k < 100):
        for i in range(10):
            #print(k, i)
            df = pd.read_csv('project4_data_part{}.csv'.format(i+1))
        
            y = df['y']
            
            X = df[set(df.columns) - set('y')]
            
            ## standardize all features
            scaler = StandardScaler().fit(X)
            X_std = scaler.transform(X)
            X_std = np.concatenate((np.ones((X.shape[0], 1)), X_std), axis=1)
        
            betai = beta[i,:]
            ui = u[i, :]
            
            beta_new[i,:] = scipy.optimize.minimize(
            target_beta,
            betai,
            args=(ui, b, rho, y, X_std),
            method='L-BFGS-B').x
            
        k = k + 1
        
        li.append(b)
        
        # update average of beta
        b = np.mean(beta_new, axis = 0)
        
        # update u: 
        r = beta_new - np.ones(10).reshape(10,1) * b
        
        u = u + r
        
        #update beta:
        beta = beta_new
        
        stop = np.sum(r**2)
                
        print(rho, k, b)
    
    pd.DataFrame(li).to_csv('results_rho{}.csv'.format(rho))  
    
    
    
    