# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 19:34:05 2024

@author: lilachhnb10
"""


import numpy as np
from sympy import exp, log, LambertW
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


#%% life history data

# change path to data file directory
path = r"C:/Users/lilachhnb10/Downloads/41586_2014_BFnature12789_MOESM42_ESM.xls"

life_tables = pd.read_excel(path, [x for x in range(1, 90)])
print(life_tables[89])


#%% model functions


def l_x(x, M, G, R):
    return np.exp((-G/R)*(np.exp(R*x)-1)-M*x)


def mortality(x, M, G, R):
    return G*np.exp(R*x)+M


def death(R, G, M, A):
    return (G - M*LambertW(G*exp((G + log(1/A)*R)/M)/M) + log(1/A)*R)/(M*R)


#%% yellow baboon


yellow_baboon_df = life_tables[37]

xx = np.array(yellow_baboon_df['Yellow baboon'][6:29].array)
yy = np.array(yellow_baboon_df[5][6:29].array)

m = np.array(yellow_baboon_df['VertMammal'][6:29].array).mean()

popt, pcov = curve_fit(mortality, xx, yy, bounds=(0.00001, [1., 1., 5.]))

M, G, R = popt

qx = [mortality(x, M, G, R) for x in xx] 

plt.plot(xx, yy)
plt.plot(xx, qx)

m *= 1 - l_x(5, M, G, R) + (1-0.209)*(1-0.1116)*(1-0.0634)*(1-0.0312) 

#%% yellow baboon fertility


fertility = np.array(yellow_baboon_df['VertMammal'][6:29].array)
plt.plot(xx, fertility)

#%% lion


lion_df = life_tables[36]

xx = np.array(lion_df['Lion'][3:21].array)
yy = np.array(lion_df['LTPeriod'][3:21].array)

m=np.array(lion_df['years'][3:21].array).mean()

popt, pcov = curve_fit(mortality, xx, yy, bounds=(0.00001, [1., 1., 5.]))

M, G, R = popt

qx = [mortality(x, M, G, R) for x in xx] 

plt.plot(xx, yy)
plt.plot(xx, qx)

m *= 1 - l_x(2, M, G, R) + 534/2643


#%% lion fertility

fertility = np.array(lion_df['years'][3:21].array)
plt.plot(xx, fertility)

#%% killer whale


killer_whale_df = life_tables[32]

xx = np.array(killer_whale_df['Killer whale'][12:35].array)
yyy = np.array(killer_whale_df['Orcinus orca'][12:36].array)

yy = np.zeros((23))

for n in range(23):
    if n!=23:
        yy[n] = (yyy[n]-yyy[n+1])/yyy[n]

m = np.array(killer_whale_df[11][12:36].array).mean()

popt, pcov = curve_fit(mortality, xx, yy, bounds=(0.00001, [0.2, 0.2, 2.]))

M, G, R = popt

qx = [mortality(x, M, G, R) for x in xx] 

plt.plot(xx, yy)
plt.plot(xx, qx)

m *= 1 - l_x(11, M, G, R) + 794.3/979.6

#%% killer whale fertility

fertility = np.array(killer_whale_df[11][12:35].array)

plt.plot(xx, fertility)



#%% human


japan_df = life_tables[23]

xx = np.array(japan_df['Swedes born in 1881'][14:107].array)
yy = np.array(japan_df[13][14:107].array)

m = np.array(japan_df['Human'][14:107].array).mean()

popt, pcov = curve_fit(mortality, xx, yy, bounds=(0.00001, [0.2, 0.2, 2.]))

M, G, R = popt

qx = [mortality(x, M, G, R) for x in xx] 

plt.plot(xx, yy)
plt.plot(xx, qx)

m *= 1 - l_x(13, M, G, R) + 0.79788


#%% human fertility

fertility = np.array(japan_df['Human'][14:107].array)
plt.plot(xx, fertility)




