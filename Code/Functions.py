# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:21:30 2019

@author: david.saltiel
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import sys
import cma.purecma as pcma
from scipy.stats import multivariate_normal
from math import log
from os.path import dirname, join
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

#%%


class Test_Functions(object):
    def __init__(self):
        module_path = dirname(__file__)
        path_plot = join(module_path,'..','Plot')
        self.module_path = module_path
        self.path_plot = path_plot
    '''
    Test Functions  :
    '''

    
    def Rastrigin(self,X):
        '''
        rastrigin
        f* = 0 x*=(0 , 0)
        '''
        A = 10
        if type(X) == list:
            X = np.array(X)
            n = X.shape[0]
            X = X.reshape([1,n])
        else:
            try:
                n = X.shape[1]
            except:
                n = len(X)
                X = np.array(X).reshape([1,n])
            
        return np.sum(X**2 - A * np.cos(2 * np.pi * X), 1) + A * n
    
    
    def Cone(self,X):
        '''
        Cone :
        f* = 0 x*=(0 , 0)
        '''
        if type(X) == list:
            X = np.array(X)
            n = X.shape[0]
            X = X.reshape([1,n])
        else:
            try:
                n = X.shape[1]
            except:
                n = len(X)
                X = np.array(X).reshape([1,n])
            
        return (np.sum(X**2,1))**(1/2)
    
    
    
    def Schwefel1(self,X):
        '''
        schwefel1
        f* = 0 x*=(420.9687 , 420.9687)
        '''
        if type(X) == list:
            X = np.array(X)
            n = X.shape[0]
            X = X.reshape([1,n])
        else:
            try:
                n = X.shape[1]
            except:
                n = len(X)
                X = np.array(X).reshape([1,n])
        if np.max(X)>500 or np.min(X)<-500:
            return 500
        A = 418.9829*X.shape[1]
        return A - np.sum(X*np.sin(np.abs(X)**(1/2)),1)
    
    
    def Schwefel2(self,X):
        '''
        schwefel2
        f* = 0 x*=(0 , 0)
        '''
        if type(X) == list:
            X = np.array(X)
            n = X.shape[0]
            X = X.reshape([1,n])
        else:
            try:
                n = X.shape[1]
            except:
                n = len(X)
                X = np.array(X).reshape([1,n])
        return np.sum(np.abs(X),1)+np.prod(np.abs(X),1)
    
    def Rastrigin_dim1(self,x):
        return 10 + x**2-10*np.cos(2*np.pi*x)
    
    def Eggholder(self,X):
        '''
        eggholder
        f* = -959.6407 x*=(512 , 404.2319)
        '''
        if type(X) == list:
            X = np.array(X)
            n = X.shape[0]
            X = X.reshape([1,n])
        else:
            try:
                n = X.shape[1]
            except:
                n = len(X)
                X = np.array(X).reshape([1,n])
        if np.max(X)>512 or np.min(X)<-512:
            return 500
        return -(X[:,1]+47)*np.sin(np.abs(X[:,0]/2+X[:,1]+47)**0.5)-X[:,0]*np.sin(np.abs(X[:,0]-X[:,1]-47)**0.5)


        
    
    def compute_min(self,list_param,n,tol1,tol2,func,option, choice_prior, verbose) :
        '''
        g : objective function g
        G : cdf of g
        f : distribution of the min of g
        '''
        
        '''
        prior : mu,sigma^2 -> NIG(mu_0,v,a,b)
        ==> E(mu)=mu_0 and E(sigma^2)=b/(a-1) (for a>1)
        1) simulate X -> N(E(mu),E(sigma^2))
        2) update mu and sigma^2 and get new E(mu) and E(sigma^2):
            E(mu) = (v*mu_0+n*x_bar)/(v+n)
            E(sigma^2) = B/(a+n2-1) where B = b+1/2*n*sigma_bar^2+(n*v)/(v+n)/2*(x_bar-mu_0)**2
        3) repeat 1)
        '''
        list_x_star = []
        list_fx_star = []
       
        variance = np.ones([n,n])
        mu_0, k_0, v_0, psi, factor = list_param[0], list_param[1], \
            list_param[2], list_param[3], list_param[4]
        x_star_min = mu_0
        var_star_min = psi
        fx_star_min = sys.float_info.max
        count = 0
        step = 0
        dim = mu_0.shape[0]
        initial_option = option
        
        while np.linalg.norm(variance) > tol1:
            if choice_prior == 'NIW' :
                E_mu = mu_0
                E_sigma = psi/(v_0-n-1)
            else :
                E_mu = mu_0
                E_sigma = psi/v_0
            step += 1.0
            
            # prior
            sample_X = np.random.multivariate_normal(mean=E_mu, 
                        cov=E_sigma, size=(n), check_valid ='ignore')
    
            g_x = func(sample_X)
            df = pd.DataFrame(sample_X)
            df['g_x'] = g_x
            try:
                df['d'] = multivariate_normal.pdf(sample_X, mean=E_mu, cov=E_sigma,
                  allow_singular=True)
            except:
                df['d'] = np.log(np.arange(n)+2)
                df['d'] = df['d'] / sum(df['d'])
            
            x_order_d = df.sort_values( by=['d'],ascending  = False)
            x_order_f = x_order_d.sort_values( by=['g_x'],kind='mergesort')
    
            d_ordered = x_order_d[x_order_d.columns[-1:]].values
            d_ordered = d_ordered / sum(d_ordered)
            x_order_f = x_order_f[x_order_f.columns[:-2]].values
            x_order_d = x_order_d[x_order_d.columns[:-2]].values
            x_all = df[df.columns[:-2]].values
            d = df['d'][:, None] 
            d = d / sum(d)
            
            lam = int(n / 2 + 1)
            d_m = d[:lam]
            d_m = d_m / sum(d_m)
            d_ordered_m = d_ordered[:lam]
            d_ordered_m = d_ordered_m / sum(d_ordered_m)
            
            if option == 'StrategyOne':
                x_bar_f = np.sum(x_order_f[:lam]*d_ordered_m,axis=0)
                x_bar = np.sum(x_all[:lam]*d_m,axis=0)
                acc = 1 
                # check that all the scalar products are positive
                for i in range(2,n):
                    if np.dot(x_order_f[i]-x_order_f[1],x_order_f[1]-x_order_f[0]) < 0:
                        break
                if i == n:
                    acc = 2
                    mean = x_bar_f - (x_bar-E_mu)
                    mean = E_mu + (x_bar_f-x_bar) * acc
                else:
                    mean = x_bar_f - (x_bar-E_mu)
                
            else:
                x_bar_f = np.sum(x_order_f * d_ordered, axis=0)
                x_bar = np.sum(x_all * d, axis=0)
                #acc =  1 + math.log(step*10)
                acc = 1 
                
                # check that all the scalar products are positive
                for i in range(2,n):
                    if np.dot(x_order_f[i]-x_order_f[1],x_order_f[1]-x_order_f[0]) < 0:
                        break
                    
                if i == n:
                    acc = 2
                    mean = E_mu + (x_order_f[0]-E_mu) * acc
                else:
                    mean = x_order_f[0]
    
            if step % 20 == 0 and verbose:
                print( 'step:',step, 'fmin:',  fx_star_min)
    
            # sigma part :
            sigma_emp = np.dot((x_all-x_bar).T,(x_all-x_bar)*d)
            sigma_ordered = np.dot((x_order_f-x_bar_f).T, (x_order_f-x_bar_f)
                                * d_ordered)
            # variance = np.dot(np.dot(E_sigma, np.linalg.inv(sigma_emp)), sigma_ordered)
            # other : 
            variance = sigma_ordered - (sigma_emp-E_sigma)
            
            variance_norm = np.linalg.norm(variance)
            fx_star = func(mean)
    
            if fx_star < fx_star_min:
                if fx_star_min - fx_star < tol2:
                    print('termination by {fx_star_min - fx_star < tol2}')
                    return list_x_star, list_fx_star
                count = 0
                option = initial_option
                factor = 1
                fx_star_min = fx_star
                x_star_min = mean
                if variance_norm < 100 * n:
                    var_star_min = variance
                list_x_star.append(mean)
                list_fx_star.append(func(mean))
            else:
                list_x_star.append(x_star_min)
                list_fx_star.append(func(x_star_min))
                count = count + 1
                if option == 'StrategyOne':
                    mean = x_star_min
                elif count %2 == 0:
                     mean = x_star_min
                
                if variance_norm > 100 * n:
                    mean = x_star_min
                    variance = var_star_min 
                    mu_0 = mean
                factor = 2**dim
                
                threshold = 3
                if count == threshold-1:
                    factor = 4**dim
    
                if count == threshold:
                    mean = x_star_min
                    variance = var_star_min 
                    mu_0 = mean
                if count >= threshold:
                    factor = (1/4)**dim
                if count > 3 * threshold:
                    factor = (1/5)**dim
                if count > 4 * threshold:
                    factor = (1/10)**dim
    
                if count > 30:
                    print('termination by {var contraction exit}')
                    return list_x_star, list_fx_star
                
            #print(step," mean=", mean," fx_star=", fx_star, " factor=",factor, " count=", count)
            #print("variance =", variance)
            mu_0 = (mu_0 * k_0 + n * mean)/(k_0+n)
            k_0 = k_0 + n
            v_0 = v_0 + n
            psi = psi + (k_0*n)/(k_0+n)*np.dot(np.mat(mean-mu_0),np.mat(mean-mu_0).T) + variance*(n-1)
            psi = psi * factor
            #print('condition on norm of variance ', variance_norm)
        print('termination by {var exit}')
        return list_x_star, list_fx_star



    def compare(self,func, mu_0, min_f, choice_prior, verbose):
        tol1 = 1e-30
        tol2 = 1e-30
        dim = len(mu_0)
        mu_0 = np.array(mu_0)
        n =  4 + int(3 * log(dim))
    #    mu_0 = np.array([400,400])
        k_0 = 4
        v_0 = n + 2
        factor = 1
        psi = np.eye(dim)
        list_param = [mu_0, k_0, v_0, psi, factor]
    
        np.random.seed(3)
        list_x_star_1, val1 = self.compute_min(list_param, n, tol1, tol2, func,
                                          'StrategyOne', choice_prior, verbose)
        np.random.seed(3)
        list_x_star_2, val2 = self.compute_min(list_param, n, tol1, tol2, func,
                                          'StrategyTwo', choice_prior, verbose)
        
        random.seed(3)
        res = pcma.fmin(func, mu_0, 1, verb_disp=1000)
        data = res[1].logger._data
        cma_esvalues = [f for f in data['fit']]
        cma_esvalues = cma_esvalues[1:]
        best_seen = cma_esvalues[0]
        cma_bestvalues = []
        for elem in cma_esvalues:
            if elem < best_seen:
                cma_bestvalues.append(elem)
                best_seen = elem
            else:
                cma_bestvalues.append(best_seen)
        
        val1 = np.abs([i - min_f for i in val1])
        val2 = np.abs([i - min_f for i in val2])
        cma_bestvalues = np.abs([i - min_f for i in cma_bestvalues])
           
        plt.figure(figsize=(6,4.5))
        plt.semilogy(np.arange(len(val2)),val2,label='B-CMA-ES S2')
        plt.semilogy(val1,label='B-CMA-ES S1')
        plt.semilogy(cma_bestvalues, label ='CMAES')
        plt.title(func.__name__ + ' convergence using ' + choice_prior + ' prior')
        plt.xlabel('iteration steps')
        plt.ylabel('error')
        plt.legend()
        plt.savefig(self.path_plot+'\\convergence_' + func.__name__ + '_' + choice_prior + '.png')
        plt.show()
    


    def plot_(self,function,m):
        func_name = function.__name__
        fig = plt.figure(figsize=(6,4.5))
        ax = fig.gca(projection='3d')
        
        # Make data.
        X = np.arange(-m, m, 0.25)
        Y = np.arange(-m, m, 0.25)
        X, Y = np.meshgrid(X, Y)
        
        if func_name == 'Rastrigin' :
            Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
            (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
        if func_name == 'Cone' :
            Z = (X**2 + Y**2)**0.5
        if func_name == 'Schwefel1' :
            Z = 418.9829*2 - X*np.sin(abs(X)**0.5) - Y*np.sin(abs(Y)**0.5)
        
        if func_name == 'Schwefel2' :
            Z = abs(X) + abs(Y) + abs(X) * abs(Y) 
        
        if func_name == 'Eggholder':
            Z = -(Y+47)*np.sin(abs(X/2+Y+47)**0.5)-X*np.sin(abs(X-Y-47)**0.5)
            
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                               linewidth=0, antialiased=False)
        
        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(5))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(self.path_plot+'\\'+'Functions\\'+func_name+'.png')
        plt.show()