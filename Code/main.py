# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:41:39 2019

@author: david.saltiel
"""
#%%
from Functions import Test_Functions
#%%
test = Test_Functions()
#function = test.Rastrigin
#function = test.Cone
#function = test.Schwefel1
#function = test.Schwefel2
function = test.Eggholder
#%%


if function.__name__ in ['Rastrigin','Cone', 'Schwefel2'] :
    mu = 2 * [10]
    min_f = 0
    if function.__name__ == 'Cone' :
        m = 100
    else :    
        m = 10
    

if function.__name__ == 'Eggholder':
    mu = [500, 400]
    min_f = -959.6407
    m = 550

if function.__name__ == 'Schwefel1':
    mu = [400, 400]
    min_f = 0
    m = 500
    

test.compare(function, mu, min_f, 'NW', False)
test.compare(function, mu, min_f, 'NIW', False)
test.plot_(function,m)

