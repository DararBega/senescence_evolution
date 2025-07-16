# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:00:17 2023

@author: jojo
"""

#%% imports


import numpy as np
from scipy import integrate
from sympy import exp, log, LambertW
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import fsolve, brentq

import warnings
warnings.filterwarnings("ignore")


#%% model functions


def mortality(age, M, G, R):
    return M+G*np.exp(R*age)


def survival(age, M, G, R):
    return np.exp((-G/R)*(np.exp(R*age)-1)-M*age)


def death(R, G, M, A):
    return (G - M*LambertW(G*exp((G + log(1/A)*R)/M)/M) + log(1/A)*R)/(M*R)


def LRS_fitness_funk(age, M, G, R, m):
    return np.exp((-G / R) * (np.exp(R * age) - 1) - M * age) * m


def LRS_fitness(M, G, R, m, x_s, A):
    x_d = death(R, G, M, A)
    return integrate.quad(LRS_fitness_funk, x_s, x_d, agrs=(M, G, R, m))[0]


def generation(age, M, G, R, m):
    return age*np.exp((-G/R)*(np.exp(R*age)-1)-M*age)*m


def fit_funk (age, R, M, G, r, m):
    return np.exp(-r*age)*survival(age, M, G, R)*m


def neglectable_death(G, M, A):
    return -log(A)/(G + M)


def DD_maternity(M, G, R, x_s, A):
    x_d = death(R, G, M, A)
    m = 1/integrate.quad(survival, x_s, x_d, args=(M, G, R))[0]
    return m
    

def generation_t(M, G, R, x_s, A):
    x_d = death(R, G, M, A)
    m = DD_maternity(M, G, R, x_s, A)
    return integrate.quad(generation, x_s, x_d, args=(M, G, R, m))[0]


def fit_funk_integ(r, M, G, R, m, x_s, A):
    x_d = death(R, G, M, A)
    return integrate.quad(fit_funk, x_s, x_d, args=(R, M, G, r, m))[0]-1


def fitness(M, G, R, m, x_s, A):
    x_d = death(R, G, M, A)
    try:
        r = brentq(fit_funk_integ, -5, 5, args=(M, G, R, m, x_s, A), maxiter=400)
        sol, dev = integrate.quad(fit_funk, x_s, x_d, args=(R, M, G, r, m))
        if 1-0.01<sol<1+0.01:
            return r
        else:
            r = fsolve(fit_funk_integ, x0=0, args=(M, G, R, m, x_s, A))
            sol, dev = integrate.quad(fit_funk, x_s, x_d, args=(R, M, G, r, m))
            if 1-0.01<sol<1+0.01:
                return float(r)
            else:
                return np.nan
    except:
        r = fsolve(fit_funk_integ, x0=0, args=(M, G, R, m, x_s, A))
        sol, dev = integrate.quad(fit_funk, x_s, x_d, args=(R, M, G, r, m))
        if 1-0.01<sol<1+0.01:
            return float(r)
        else:
            return np.nan

def fit_x_s_DD(x_s, M, G, R, m, A):
    x_d = death(R, G, M, A)
    try:
        r = brentq(fit_funk_integ, -5, 5, args=(M, G, R, m, x_s, A), maxiter=400)
        sol, dev = integrate.quad(fit_funk, x_s, x_d, args=(R, M, G, r, m))
        if 1-0.01<sol<1+0.01:
            return r
        else:
            r = fsolve(fit_funk_integ, x0=0, args=(M, G, R, m, x_s, A))
            sol, dev = integrate.quad(fit_funk, x_s, x_d, args=(R, M, G, r, m))
            if 1-0.01<sol<1+0.01:
                return float(r)
            else:
                return -99999
    except:
        r = fsolve(fit_funk_integ, x0=0, args=(M, G, R, m, x_s, A))
        sol, dev = integrate.quad(fit_funk, x_s, x_d, args=(R, M, G, r, m))
        if 1-0.01<sol<1+0.01:
            return float(r)
        else:
            return -99999


def DD_external2(M, G, R, m, x_s, A):
    M_dd = fitness(M, G, R, m, x_s, A) + M
    if (M_dd >= 0.0000000001) and (M_dd <= 0.9999999999):
        return M_dd
    elif M_dd > 0.9999999999:
        return 0.9999999999
    else:
        return 0.0000000001
    

def DD_external(M, G, R, m, x_s, A):
    M_dd = fitness(M, G, R, m, x_s, A) + M
    if (M_dd >= 0.0000000001):
        return M_dd
    else:
        return 0.0000000001
    

def DD_xs(M, G, R, m, A):
    x_d = death(R, G, M, A)
    x_s_dd = brentq(fit_x_s_DD, 0, x_d, args=(M, G, R, m, A), maxiter=400)
    return x_s_dd


def MA_LRS5(M, G, m, R, A):
    x_d = death(R, G, M, A)
    x_s = DD_xs(M, G, R, m, A)
    if x_d > x_s and G!=0:     
        return integrate.quad(funk_LRS, x_s, x_d, args=(M, G, R, m))[0]
    elif x_d > x_s or G==0:
        return 0
    else:
        return np.nan


def fitness_stable(M, G, R, m, x_s, A):
    x_d = death(R, G, M, A)
    LRS = integrate.quad(fit_funk, x_s, x_d, args=(R, M, G, 0, m))[0]
    return LRS


def fitfunk_stable(age, M, G, R, m):
    return np.exp((-G/R)*(np.exp(R*age)-1)-M*age)*m


def fitness_stable2(M, G, R, m, x_s, A):
    x_d = death(R, G, M, A)
    r = integrate.quad(fitfunk_stable, x_s, x_d, args=(M, G, R, m))[0]
    return r


def MA(M, G, R, m, x_s, d, A):
    return (fitness(M, G, R+d, m, x_s, A)-fitness(M, G, R, m, x_s, A))/d


def deliminator(age, R, M, G, r, m):
    return ((G/R**2)*np.exp(R*age)-G/R**2-(G/R)*age*np.exp(R*age))*np.exp(-r*age)*np.exp((-G/R)*(np.exp(R*age)-1)-M*age)*m
    

def denominator(age, R, M, G, r, m):
    return age*np.exp(-r*age)*np.exp((-G/R)*(np.exp(R*age)-1)-M*age)*m


def generation_time(R, M, G, r, m, x_s, A):
    x_d = death(R, G, M, A)
    return integrate.quad(denominator, x_s, x_d, args=(R, M, G, R, m))[0]


def MA2(R, M, G, r, m, x_s, A):
    x_d = death(R, G, M, A)
    if x_d > x_s and G!=0:
        return integrate.quad(deliminator, x_s, x_d, args=(R, M, G, r, m), limit=400)[0]/integrate.quad(denominator, x_s, x_d, args=(R, M, G, r, m), limit=400)[0]
    elif x_d > x_s and G==0:
        return 0
    else:
        print('!!!', x_d, '<',x_s)
        return np.nan


def funk_LRS(age, M, G, R, m):
    return ((G/R**2)*np.exp(R*age)-G/R**2-(G/R)*age*np.exp(R*age))*np.exp((-G/R)*(np.exp(R*age)-1)-M*age)*m
 
   
def MA_LRS(M, G, R, x_s, A):
    x_d = death(R, G, M, A)
    if x_d > x_s and G!=0:     
        m = DD_maternity(M, G, R, x_s, A)
        return integrate.quad(funk_LRS, x_s, x_d, args=(M, G, R, m))[0]
    elif x_d > x_s and G==0:
        return 0
    else:
        return np.nan


def MA_LRS3(M, G, R, m, x_s, A):
    x_d = death(R, G, M, A)
    return integrate.quad(funk_LRS, x_s, x_d, args=(M, G, R, m))[0]


def MA_LRS2(M, G, R, m, x_s, A):
    x_d = death(R, G, M, A)
    return integrate.quad(funk_LRS, x_s, x_d, args=(M, G, R, m))[0]/(m*integrate.quad(generation, x_s, x_d, args=(M, G, R))[0])


def MA2_LRS(M, G, R, m, x_s, d, A):
    return (fitness_stable(M, G, R+d, m, x_s, A)-fitness_stable(M, G, R, m, x_s, A))/d 
    

def d_sel_d_r(M, G, R, m, x_s, A):
    h = 1e-4
    R1 = fitness(M, G, R+h, m, x_s, A)
    R2 = fitness(M, G, R-h, m, x_s, A)
    if (MA2(R1, M, G, R+h, m, x_s, A)-MA2(R2, M, G, R-h, m, x_s, A)) > 0:
        return True
    else:
        return False


def a_survival(age, M, G, R, a, R2):
    if age<a:
        return np.exp((-G/R)*(np.exp(R*age)-1)-M*age) - 0.01
    else:
        return np.exp( (-G/R)*(np.exp(R*a)-1) - ((G*np.exp(a*(R-R2)))/R2)*(np.exp(R2*age)-np.exp(R2*a)) - M*age ) - 0.01


def a_MA(R, M, G, r, m, x_s, a, A):
    x_d = death(R, G, M, A)
    return integrate.quad(deliminator, a, x_d, args=(R, M, G, r, m))[0]/integrate.quad(denominator, x_s, x_d, args=(R, M, G, r, m))[0]    


def a_MA_LRS(M, G, R, m, a, A):
    x_d = death(R, G, M, A)
    return integrate.quad(funk_LRS, a, x_d, args=(M, G, R, m))[0]


def age_of_death(M, G, R, a, R2):
    sol = brentq(a_survival, 0, 1000, args=(M, G, R, a, R2))
    return sol
    

def neglectable_LRS(M, G, x_s):
    return -G*(2+2*(M+G)*x_s**2+((M+G)*x_s)**2)/(2*(M+G)**2) 


def neglectable_EL(G, x_s, m):
    L = float(LambertW(x_s*m))
    return -G*x_s*(2+2*L+L**2)/(2*L*(1+L))


def x_s_EU_selection(M, G, R, m, x_s, A):
    r = fitness(M, G, R, m, x_s, A)
    delim = - np.exp(- r * x_s) * survival(x_s, M, G, R) * m
    T = generation_time(R, M, G, r, m, x_s, A)
    return  delim / T


def x_s_LRS_selection(x_s, M, G, R, A):
    return -survival(x_s, M, G, R) * DD_maternity(M, G, R, x_s, A)


def roots_inrisk_peak(M, x_s):
    roots = np.roots([x_s**2, 3*M*x_s**2, 2*M*x_s - 2 + 3*M**2*x_s**2, M**3*x_s**3 + 2*M**2*x_s + 2*M])
    real_positive = []
    roots_sel = []
    if len(roots[np.isreal(roots)].real)==3:
        for root in roots:
            if root>0:
                real_positive.append(root)
                roots_sel.append(neglectable_LRS(M, root, x_s))
        return real_positive[roots_sel.index(min(roots_sel))]
    else:
        return np.nan


def dd_G_sel_funk(age, M, G, R, m):
    return (-(1/R)*(np.exp(R*age)-1))*np.exp((-G/R)*(np.exp(R*age)-1)-M*age)*m
 
   
def dd_G_sel(M, G, R, x_s, A):
    x_d = float(death(R, G, M, A))
    if x_d > x_s:     
        m = DD_maternity(M, G, R, x_s, A)
        dd_G = integrate.quad(dd_G_sel_funk, x_s, x_d, args=(M, G, R, m))[0]
        return dd_G
    else:
        return np.nan


def G_deliminator(age, R, M, G, r, m):
    return (-(1/R)*(np.exp(R*age)-1))*np.exp(-r*age)*np.exp((-G/R)*(np.exp(R*age)-1)-M*age)*m
    

def G_denominator(age, R, M, G, r, m):
    return age*np.exp(-r*age)*np.exp((-G/R)*(np.exp(R*age)-1)-M*age)*m


def MA_G(R, M, G, r, m, x_s, A):
    x_d = float(death(R, G, M, A))
    if x_d > x_s and G!=0:
        return integrate.quad(G_deliminator, x_s, x_d, args=(R, M, G, r, m))[0]/integrate.quad(G_denominator, x_s, x_d, args=(R, M, G, r, m))[0]
    elif x_d > x_s and G==0:
        return 0
    else:
        return np.nan


def RG_trade_off(z, R_min, R_max, G_min, G_max, alpha, beta):
    R = (R_max - R_min) * z ** alpha + R_min
    G = (G_max - G_min) * (1 - z) ** beta + G_min
    return R, G 


def find_z(R, G, R_min, R_max, G_min, G_max, alpha, beta):
    z_R = ((R - R_min) / (R_max - R_min)) ** (1 / alpha)
    z_G = 1 - ((G - G_min) / (G_max - G_min)) ** (1 / alpha)
    return z_R, z_G


def find_G_cost_max(G_cost_min, G_max, G_min, alpha, G, benifit=651):
     return (benifit - G_cost_min) / ((1 - ((1 / (G_max - G_min)) * G - (G_min / (G_max - G_min))))**alpha) + G_cost_min



def find_R_cost_max(R_cost_min, R_max, R_min, beta, R, benifit=14):
     return (benifit - R_cost_min) / ((1 - ((1 / (R_max - R_min)) * R - (R_min / (R_max - R_min))))**beta) + R_cost_min



def Rm_trade_off(z, R_min, R_max, m_min, m_max, alpha, beta):
    R = (R_max - R_min) * z ** alpha + R_min
    m = (m_max - m_min) * (1 - (1 - z) ** beta) + m_min
    return R, m 


def dd_m_selection_funk(age, M, G, R):
    return np.exp((-G/R)*(np.exp(R*age)-1)-M*age)


def dd_selection_m(M, G, R, x_s, A):
    x_d = float(death(R, G, M, A))
    if x_d > x_s and G!=0:     
        dd_m = integrate.quad(dd_m_selection_funk, x_s, x_d, args=(M, G, R))[0]
        return dd_m
    else:
        return np.nan
    

def dd_m_sel(M, G, R, m, x_s, A):
    x_d = float(death(R, G, M, A))
    LRS = LRS_fitness(M, G, R, m, x_s, A)
    if x_d > x_s and G!=0:     
        dd_m = integrate.quad(dd_m_selection_funk, x_s, x_d, args=(M, G, R))[0] / LRS
        return dd_m
    else:
        return np.nan
    

def dLRS_dm(z, R_min, R_max, m_min, m_max, alpha, beta):
    return beta * (m_max - m_min) * (1 - z) ** (beta - 1) / ((m_max - m_min) * (1 - (1 - z) ** beta) + m_min)


def dLRS_dz(z, R_min, R_max, m_min, m_max, alpha, beta, G, M, x_s, A):
    R = (R_max - R_min) * z ** alpha + R_min
    return alpha * (R_max - R_min) * z ** (alpha - 1) * MA_LRS(M, G, R, x_s, A) + dLRS_dm(z, R_min, R_max, m_min, m_max, alpha, beta)


def m_delim(age, M, G, R, x_s, m, A):
    r = fitness(M, G, R, m, x_s, A)
    return np.exp(-r*age)*np.exp((-G/R)*(np.exp(R*age)-1)-M*age)


def m_denom(age, M, G, R, x_s, m, A):
    r = fitness(M, G, R, m, x_s, A) 
    return age*np.exp(-r*age)*np.exp((-G/R)*(np.exp(R*age)-1)-M*age)*m

def drdm(M, G, R, x_s, m, A):
    x_d = float(death(R, G, M, A))
    return integrate.quad(m_delim, x_s, x_d, args=(M, G, R, x_s, m, A))[0] / integrate.quad(m_denom, x_s, x_d, args=(M, G, R, x_s, m, A))[0] 


def find_beta(alpha, R, R_min, R_max, m, m_min, m_max):
    return np.log(1-(m-m_min)/(m_max-m_min)) / np.log(((R-R_min)/(R_max-R_min))**(1/alpha))


def find_m_max(alpha, beta, R, R_min, R_max, m, m_min):
    return m_min + (m-m_min) / (1 - (1 - ((R-R_min)/(R_max-R_min))**(1/alpha))**beta)


def cost_func(theta, specie='lion'):
    alpha = theta
    beta = 1
    R_max = 0.5
    A = 0.0001
    R_min = 0.0001
    m_min = 0
    
    trade_data = pd.DataFrame(columns=('M', '$G$', '$R$', 'x_s', 'hue', '$z$', '$dR$', '$dm$', '$dz$', '$r$', 'LRS'))
    
    # whale
    if specie == 'whale': 
        M, G, x_s, m, R_sp = 0.00001, 0.00094, 11, 0.051, 0.048
    
    # baboon
    elif specie == 'lion': 
        M, G, x_s, m, R_sp = 0.0522, 0.0025, 2, 0.2, 0.325

    # lion
    elif specie == 'baboon':
        M, G, x_s, m, R_sp = 0.05, 0.003, 5, 0.182, 0.2

    # human
    elif specie == 'human':
        M, G, x_s, m, R_sp = 0.00001, 0.00041, 13, 0.036, 0.071
    
    
    m_max = find_m_max(alpha, beta, R_sp, R_min, R_max, m, m_min)
    print(m_max)
    
    for z in np.arange(0.0001, 1, 0.01):
        R, m = Rm_trade_off(z, R_min, R_max, m_min, m_max, alpha, beta)
        r = fitness(M, G, R, m, x_s, A)
        LRS = fitness_stable(M, G, R, m, x_s, A)
        
        dR = MA_LRS(M, G, R, x_s, A) * (alpha * (R_max - R_min) * z ** (alpha - 1))
        dm = dd_selection_m(M, G, R, x_s, A) * (beta * (m_max - m_min) * (1 - z) ** (beta - 1))
        
        dz = dLRS_dz(z, R_min, R_max, m_min, m_max, alpha, beta, G, M, x_s, A)
        trade_data.loc[len(trade_data)] = [M, G, R, x_s, specie, z, dR, dm, np.abs(dz), r, LRS]
    
    R_dz = trade_data.iloc[(trade_data["$dz$"].abs()).argmin()]["$R$"]
    #R_dz = trade_data[trade_data["$dz$"]==min(trade_data['$dz$'])]["$R$"]
    cost = np.abs(R_sp - R_dz)
    return cost


def fit_alpha(specie='lion'):
    alphas = []
    costs = []
    for alpha in np.arange(1, 2, 0.1):
        cost = cost_func(alpha, specie)
        alphas.append(alpha)
        costs.append(cost)
        print(cost, alpha)
    min_cost_alpha = alphas[costs.index(min(costs))]
    
    print(min_cost_alpha)
    return min_cost_alpha
        

def plot_figure_5():
    
    R_max= 0.5
    A = 0.0001
    R_min = 0.0001
    m_min = 0

    trade_data = pd.DataFrame(columns=('M', '$G$', '$R$', 'x_s', 'species', '$z$', '$dR$', '$dm$', '$dz$', '$r$', 'LRS'))

    for M, G, R_sp, m_sp, x_s, hue, col, alpha, beta in [[0.00001, 0.00041, 0.071, 0.036, 13, 'Human', 'darkred', 1, 1], 
                                                         [0.00001, 0.0094, 0.048, 0.051, 11, 'Killer whale', 'darkblue', 1.3, 1], 
                                                         [0.05, 0.003, 0.2, 0.182, 5, 'Yellow baboon', 'gold', 1.5, 1], 
                                                         [0.0522, 0.0025, 0.325, 0.2, 2, 'Lion', 'orange', 1.5, 1]]:
        
        m_max = find_m_max(alpha, beta, R_sp, R_min, R_max, m_sp, m_min)
        print(m_max, DD_maternity(M, G, R_sp, x_s, A), fitness_stable(M, G, R_sp, m_sp, x_s, A))
        for z in np.arange(0.0001, 1, 0.01):
            
            R, m = Rm_trade_off(z, R_min, R_max, m_min, m_max, alpha, beta)
            r = fitness(M, G, R, m, x_s, A)
            LRS = fitness_stable(M, G, R, m, x_s, A)
            
            dR = MA_LRS(M, G, R, x_s, A) * (alpha * (R_max - R_min) * z ** (alpha - 1))
            dm = dd_selection_m(M, G, R, x_s, A) * (beta * (m_max - m_min) * (1 - z) ** (beta - 1))
            
            dz = dLRS_dz(z, R_min, R_max, m_min, m_max, alpha, beta, G, M, x_s, A)
            trade_data.loc[len(trade_data)] = [M, G, R, x_s, hue, z, dR, dm, dz, r, LRS]

    plt.figure()
    sns.lineplot(data=trade_data, x='$R$', y='$dz$', hue='species', palette=['darkred', 'darkblue', 'gold', 'darkorange']).set_ylim(-5, 5)
    plt.axhline(0, ls='--', c='black')
    plt.scatter([0.071, 0.048, 0.2, 0.325], [0, 0, 0, 0], s=80, marker=(5, 2), c=['darkred', 'darkblue', 'gold', 'darkorange'] ,zorder=20)
    plt.ylabel(r'$\frac{dLRS}{dz}$')
    
    return


def plot_heatmaps_for_figures_3_and_4(specie='baboon'):
    '''
    select specie: (human, whale, baboon, lion, or midle for parameters in the midle of the parameter range)
    and get heatmaps used for plots 3 and 4 
    '''

    A = 0.0001
    res = 100

    # whale
    if specie == 'whale':
        color = 'darkblue'
        M_max, G_max, x_s_max, m_max, R_max = 0.01, 0.01, 20, 1, 0.5 
        M_sp, G_sp, x_s_sp, m_sp, R_sp = 0.00001, 0.00094, 11, 0.051, 0.048
    
    # baboon
    elif specie == 'baboon':
        color = 'gold'
        M_max, G_max, x_s_max, m_max, R_max = 0.1, 0.01, 20, 1, 0.5 
        M_sp, G_sp, x_s_sp, m_sp, R_sp = 0.0522, 0.0025, 2, 0.257, 0.2

    # lion
    elif specie == 'lion':
        color = 'orange'
        M_max, G_max, x_s_max, m_max, R_max = 0.1, 0.01, 20, 1, 0.5 
        M_sp, G_sp, x_s_sp, m_sp, R_sp = 0.05, 0.003, 5, 0.157, 0.325

    # human
    elif specie == 'human':
        color = 'darkred'
        M_max, G_max, x_s_max, m_max, R_max = 0.01, 0.01, 20, 1, 0.5 
        M_sp, G_sp, x_s_sp, m_sp, R_sp = 0.00001, 0.00041, 14, 0.035, 0.071

    # midle
    else:
        color = 'grey'
        M_max, G_max, x_s_max, m_max, R_max = 0.1, 0.1, 20, 1, 0.1 
        M_sp, G_sp, x_s_sp, m_sp, R_sp = 0.05, 0.05, 10, 0.5, 0.05

    df = pd.DataFrame(columns=['$R$', '$M$', '$G$', '$x_s$', '$ζ$', 'dR_LRS', 'dR_EL', 'dM_LRS', 'dG_LRS', 'dG_EL', 'dx_s_LRS', 'dx_s_EL', 'dm_EL', 'heat'])

    for R in np.arange(0.0001, R_max, R_max/res):
        print(R/0.1)
        for M in np.arange(0.0001, M_max, M_max/res):
                
            dR_LRS = MA_LRS(M, G_sp, R, x_s_sp, A)
            #dM_LRS = -generation_t(M, G_sp, R, x_s_sp, A)
            # r = fitness(M, G_sp, R, m_sp, x_s_sp, A) 
            # EL = MA2(R, M, G_sp, r, m_sp, x_s_sp, A)
            
            
            #df.loc[len(df)] = [R, M, G_sp, x_s_sp, m_sp, dR_LRS, None, dM_LRS, None, None, None, None, None, 1]
            df.loc[len(df)] = [R, M, G_sp, x_s_sp, m_sp, dR_LRS, None, None, None, None, None, None, None, 1]
        
        for G in np.arange(0, G_max, G_max/res):
            
            dR_LRS = MA_LRS(M_sp, G, R, x_s_sp, A)
            #dG_LRS = dd_G_sel(M_sp, G, R, x_s_sp, A)
            
            r = fitness(M_sp, G, R, m_sp, x_s_sp, A) 
            dR_EL = MA2(R, M_sp, G, r, m_sp, x_s_sp, A)
            #dG_EL = MA_G(R, M_sp, G, r, m_sp, x_s_sp, A)
            
            
            #df.loc[len(df)] = [R, M_sp, G, x_s_sp, m_sp, dR_LRS, dR_EL, None, dG_LRS, dG_EL, None, None, None, 2]
            df.loc[len(df)] = [R, M_sp, G, x_s_sp, m_sp, dR_LRS, dR_EL, None, None, None, None, None, None, 2]
            
        for x_s in np.arange(0.2, x_s_max, x_s_max/res):
            
            dR_LRS = MA_LRS(M_sp, G_sp, R, x_s, A)
            #dx_s_LRS = x_s_LRS_selection(x_s, M_sp, G_sp, R)
            r = fitness(M_sp, G_sp, R, m_sp, x_s, A) 
            dR_EL = MA2(R, M_sp, G_sp, r, m_sp, x_s, A)
            #dx_s_EL = x_s_EU_selection(M_sp, G_sp, R, m_sp, x_s, A)
            
            
            #df.loc[len(df)] = [R, M_sp, G_sp, x_s, m_sp, dR_LRS, dR_EL, None, None, None, dx_s_LRS, dx_s_EL, None, 3]
            df.loc[len(df)] = [R, M_sp, G_sp, x_s, m_sp, dR_LRS, dR_EL, None, None, None, None, None, None, 3]
            
        for m in np.arange(0.0001, m_max, m_max/res):
            
            # LRS = MA_LRS(M_sp, G_sp, R, x_s_sp, A)
            r = fitness(M_sp, G_sp, R, m, x_s_sp, A) 
            dR_EL = MA2(R, M_sp, G_sp, r, m, x_s_sp, A)
            #dm_EL = drdm(M_sp, G_sp, R, x_s_sp, m, A)
            
            #df.loc[len(df)] = [R, M_sp, G_sp, x_s_sp, m, None, dR_EL, None, None, None, None, None, dm_EL, 4]
            df.loc[len(df)] = [R, M_sp, G_sp, x_s_sp, m, None, dR_EL, None, None, None, None, None, None, 4]


    # get data for fitness heatmap
    
    LRS_data_M_fit = df[df['heat'] == 1].pivot(index='$R$', columns='$M$', values='dR_LRS')

    EL_data_G_fit = df[df['heat'] == 2].pivot(index='$R$', columns='$G$', values='dR_EL')

    LRS_data_G_fit = df[df['heat'] == 2].pivot(index='$R$', columns='$G$', values='dR_LRS')

    EL_data_x_s_fit = df[df['heat'] == 3].pivot(index='$R$', columns='$x_s$', values='dR_EL')

    LRS_data_x_s_fit = df[df['heat'] == 3].pivot(index='$R$', columns='$x_s$', values='dR_LRS')

    EL_data_m_fit = df[df['heat'] == 4].pivot(index='$R$', columns='$ζ$', values='dR_EL')

    # get transform fitness data

    T_LRS_data_M_fit = np.log10(np.abs(LRS_data_M_fit)+1)*np.sign(LRS_data_M_fit)

    #T_EL_data_G_fit  = np.log10(np.abs(EL_data_G_fit)+1)*np.sign(EL_data_G_fit)

    T_LRS_data_G_fit = np.log10(np.abs(LRS_data_G_fit)+1)*np.sign(LRS_data_G_fit)

    #T_EL_datax_s_fit = np.log10(np.abs(EL_data_x_s_fit)+1)*np.sign(EL_data_x_s_fit)

    T_LRS_data_x_s_fit = np.log10(np.abs(LRS_data_x_s_fit)+1)*np.sign(LRS_data_x_s_fit)

    #T_EL_dat_m_fit = np.log10(np.abs(EL_data_m_fit)+1)*np.sign(EL_data_m_fit)

    # get data for senescence rate feedback

    LRS_data_M = -1 * df[df['heat'] == 1].pivot(index='$R$', columns='$M$', values='dR_LRS').diff(axis=0, periods=-1) / (M_max/res)

    EL_data_G = df[df['heat'] == 2].pivot(index='$R$', columns='$G$', values='dR_EL').diff(axis=0) / (G_max/res)

    LRS_data_G = -1 * df[df['heat'] == 2].pivot(index='$R$', columns='$G$', values='dR_LRS').diff(axis=0, periods=-1) / (G_max/res)

    EL_data_x_s = df[df['heat'] == 3].pivot(index='$R$', columns='$x_s$', values='dR_EL').diff(axis=0) / (x_s_max/res)

    LRS_data_x_s = df[df['heat'] == 3].pivot(index='$R$', columns='$x_s$', values='dR_LRS').diff(axis=0) / (x_s_max/res)

    EL_data_m = df[df['heat'] == 4].pivot(index='$R$', columns='$ζ$', values='dR_EL').diff(axis=0) / (m_max/res)

    # get data for mixed feedback 

    LRS_data_M_mix = -1 * df[df['heat'] == 1].pivot(index='$R$', columns='$M$', values='dR_LRS').diff(axis=1, periods=-1) / (M_max/res)

    EL_data_G_mix = df[df['heat'] == 2].pivot(index='$R$', columns='$G$', values='dR_EL').diff(axis=1) / (G_max/res)

    LRS_data_G_mix = -1 * df[df['heat'] == 2].pivot(index='$R$', columns='$G$', values='dR_LRS').diff(axis=1, periods=-1) / (G_max/res)

    EL_data_x_s_mix = df[df['heat'] == 3].pivot(index='$R$', columns='$x_s$', values='dR_EL').diff(axis=1) / (x_s_max/res)

    LRS_data_x_s_mix = df[df['heat'] == 3].pivot(index='$R$', columns='$x_s$', values='dR_LRS').diff(axis=1) / (x_s_max/res)

    EL_data_m_mix = df[df['heat'] == 4].pivot(index='$R$', columns='$ζ$', values='dR_EL').diff(axis=1) / (m_max/res)

    # get data for transformed senescence rate feedback

    T_LRS_data_M = np.log10(np.abs(LRS_data_M)+1)*np.sign(LRS_data_M)

    T_EL_data_G  = np.log10(np.abs(EL_data_G)+1)*np.sign(EL_data_G)

    T_LRS_data_G = np.log10(np.abs(LRS_data_G)+1)*np.sign(LRS_data_G)

    T_EL_data_x_s = np.log10(np.abs(EL_data_x_s)+1)*np.sign(EL_data_x_s)

    T_LRS_data_x_s = np.log10(np.abs(LRS_data_x_s)+1)*np.sign(LRS_data_x_s)

    T_EL_data_m = np.log10(np.abs(EL_data_m)+1)*np.sign(EL_data_m)

    # get data for transformed mixed feedback
        
    T_LRS_data_M_mix = np.log10(np.abs(LRS_data_M_mix)+1)*np.sign(LRS_data_M_mix)
    
    T_EL_data_G_mix  = np.log10(np.abs(EL_data_G_mix)+1)*np.sign(EL_data_G_mix)
    
    T_LRS_data_G_mix = np.log10(np.abs(LRS_data_G_mix)+1)*np.sign(LRS_data_G_mix)
    
    T_EL_data_x_s_mix = np.log10(np.abs(EL_data_x_s_mix)+1)*np.sign(EL_data_x_s_mix)
    
    T_LRS_data_x_s_mix = np.log10(np.abs(LRS_data_x_s_mix)+1)*np.sign(LRS_data_x_s_mix)
    
    T_EL_data_m_mix = np.log10(np.abs(EL_data_m_mix)+1)*np.sign(EL_data_m_mix)

    # plot fitness, senescence rate feedback, and mixed feedback heatmaps for selected specie

    fig1, axs = plt.subplots(2,3)

    sns.heatmap(T_LRS_data_M_fit, cbar_kws={'label': r'$LRS$'}, cmap='magma', ax=axs[0, 0], square=True, vmin=-3, vmax=0).invert_yaxis()
    sns.heatmap(EL_data_G_fit, cbar_kws={'label': r'$LRS$'}, cmap='magma', ax=axs[1, 1], square=True, vmin=-3, vmax=0).invert_yaxis()
    sns.heatmap(T_LRS_data_G_fit, cbar_kws={'label': r'$LRS$'}, cmap='magma', ax=axs[0, 1], square=True, vmin=-3, vmax=0).invert_yaxis()
    sns.heatmap(EL_data_x_s_fit, cbar_kws={'label': r'$LRS$'}, cmap='magma', ax=axs[1, 2], square=True, vmin=-3, vmax=0).invert_yaxis()
    sns.heatmap(T_LRS_data_x_s_fit, cbar_kws={'label': r'$LRS$'}, cmap='magma', ax=axs[0, 2], square=True, vmin=-3, vmax=0).invert_yaxis()
    sns.heatmap(EL_data_m_fit, cbar_kws={'label': r'$LRS$'}, cmap='magma', ax=axs[1, 0], square=True, vmin=-3, vmax=0).invert_yaxis()

    axs[0, 0].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[0, 0].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[0, 0].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[0, 0].set_xticklabels(np.arange(0, 10.5*M_max/10, M_max/10).round(decimals=5))
    axs[0, 0].scatter(M_sp*(100/M_max), R_sp*(100/R_max), c=color, ec='w', s=100) 

    axs[1, 1].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[1, 1].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[1, 1].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[1, 1].set_xticklabels(np.arange(0, 10.5*G_max/10, G_max/10).round(decimals=5))
        
    axs[0, 1].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[0, 1].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[0, 1].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[0, 1].set_xticklabels(np.arange(0, 10.5*G_max/10, G_max/10).round(decimals=5))
    axs[0, 1].scatter(G_sp*(100/G_max), R_sp*(100/R_max), c=color, ec='w', s=100)    

    axs[1, 2].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[1, 2].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[1, 2].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[1, 2].set_xticklabels(np.arange(0, 10.5*x_s_max/10, x_s_max/10).round(decimals=0))

    axs[0, 2].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[0, 2].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[0, 2].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[0, 2].set_xticklabels(np.arange(0, 10.5*x_s_max/10, x_s_max/10).round(decimals=0))
    axs[0, 2].scatter(x_s_sp*(100/x_s_max), R_sp*(100/R_max), c=color, ec='w', s=100)    

    axs[1, 0].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[1, 0].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[1, 0].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[1, 0].set_xticklabels(np.arange(0, 10.5*m_max/10, m_max/10).round(decimals=3))

    fig1.show()
    
    sns.set(font_scale=1.5)
    fig2, axs = plt.subplots(2,3)
    
    sns.heatmap(T_LRS_data_M, cbar_kws={'label': r'$±log(|\frac{∂^2LRS}{∂^2R^2}|+1)$'+"\n", 'extend':'both', 'location':'top'}, cmap='RdBu_r', ax=axs[0, 0], square=True, center=0, vmin=-6, vmax=6).invert_yaxis()
    sns.heatmap(T_EL_data_G, cbar_kws={'label': r'$\frac{∂^2r}{∂^2R^2}$'+"\n", 'extend':'both', 'location':'top'}, cmap='RdBu_r', ax=axs[1, 1], square=True, center=0, vmin=-3, vmax=3).invert_yaxis()
    sns.heatmap(T_LRS_data_G, cbar_kws={'label': r'$±log(|\frac{∂^2LRS}{∂^2R^2}|+1)$'+"\n", 'extend':'both', 'location':'top'}, cmap='RdBu_r', ax=axs[0, 1], square=True, center=0, vmin=-6, vmax=6).invert_yaxis()
    sns.heatmap(T_EL_data_x_s, cbar_kws={'label': r'$\frac{∂^2r}{∂^2R^2}$'+"\n", 'extend':'both', 'location':'top'}, cmap='RdBu_r', ax=axs[1, 2], square=True, center=0, vmin=-0.5, vmax=0.5).invert_yaxis()
    sns.heatmap(T_LRS_data_x_s, cbar_kws={'label': r'$±log(|\frac{∂^2LRS}{∂^2R^2}|+1)$'+"\n", 'extend':'both', 'location':'top'}, cmap='RdBu_r', ax=axs[0, 2], square=True, center=0, vmin=-2, vmax=2).invert_yaxis()
    sns.heatmap(T_EL_data_m, cbar_kws={'label': r'$\frac{∂^2r}{∂^2R^2}$'+"\n", 'extend':'both', 'location':'top'}, cmap='RdBu_r', ax=axs[1, 0], square=True, center=0, vmin=-0.5, vmax=0.5).invert_yaxis()
    
    axs[0, 0].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[0, 0].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[0, 0].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[0, 0].set_xticklabels(np.arange(0, 10.5*M_max/10, M_max/10).round(decimals=5))
    axs[0, 0].set_ylabel('$R$')
    axs[0, 0].set_xlabel('$M$')
    axs[0, 0].scatter(M_sp*(100/M_max), R_sp*(100/R_max), c=color, ec='w', s=100)  
        
    axs[1, 1].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[1, 1].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[1, 1].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[1, 1].set_xticklabels(np.arange(0, 10.5*G_max/10, G_max/10).round(decimals=5))
        
    axs[0, 1].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[0, 1].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[0, 1].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[0, 1].set_xticklabels(np.arange(0, 10.5*G_max/10, G_max/10).round(decimals=5))
    axs[0, 1].set_xlabel('$G$')    
    axs[0, 1].scatter(G_sp*(100/G_max), R_sp*(100/R_max), c=color, ec='w', s=100)    
    
    axs[1, 2].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[1, 2].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[1, 2].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[1, 2].set_xticklabels(np.arange(0, 10.5*x_s_max/10, x_s_max/10).round(decimals=0))
    
    axs[0, 2].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[0, 2].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[0, 2].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[0, 2].set_xticklabels(np.arange(0, 10.5*x_s_max/10, x_s_max/10).round(decimals=0))
    
    axs[1, 0].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[1, 0].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[1, 0].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[1, 0].set_xticklabels(np.arange(0, 10.5*m_max/10, m_max/10).round(decimals=3))
    
    fig2.show()    
    
    sns.set(font_scale=1.5)
    fig3, axs = plt.subplots(2,3)

    sns.heatmap(T_LRS_data_M_mix, cbar_kws={'label': r'$\frac{∂^2LRS}{∂R∂M}$'}, cmap='RdBu_r', ax=axs[0, 0],  square=True, center=0, vmin=-6, vmax=6).invert_yaxis()
    sns.heatmap(T_EL_data_G_mix, cbar_kws={'label': r'$\frac{∂^2r}{∂R∂G}$'}, cmap='RdBu_r', ax=axs[1, 1], square=True, center=0).invert_yaxis()
    sns.heatmap(T_LRS_data_G_mix, cbar_kws={'label': r'$\frac{∂^2LRS}{∂R∂G}$'}, cmap='RdBu_r', ax=axs[0, 1], square=True, center=0, vmin=-6, vmax=6).invert_yaxis()
    sns.heatmap(T_EL_data_x_s_mix, cbar_kws={'label': r'$\frac{∂^2r}{∂R∂x_s}$'}, cmap='RdBu_r', ax=axs[1, 2], square=True, center=0).invert_yaxis()
    sns.heatmap(T_LRS_data_x_s_mix, cbar_kws={'label': r'$\frac{∂^2LRS}{∂R∂x_s}$'}, cmap='RdBu_r', ax=axs[0, 2], square=True, center=0).invert_yaxis()
    sns.heatmap(T_EL_data_m_mix, cbar_kws={'label': r'$\frac{∂^2r}{∂R∂ζ}$'}, cmap='RdBu', ax=axs[1, 0], square=True, center=0).invert_yaxis()

    axs[0, 0].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[0, 0].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[0, 0].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[0, 0].set_xticklabels(np.arange(0, 10.5*M_max/10, M_max/10).round(decimals=5))
    axs[0, 0].set_ylabel('$R$')
    axs[0, 0].set_xlabel('$M$')
    axs[0, 0].scatter(M_sp*(100/M_max), R_sp*(100/R_max), c=color, ec='w', s=100)  
        
    axs[1, 1].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[1, 1].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[1, 1].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[1, 1].set_xticklabels(np.arange(0, 10.5*G_max/10, G_max/10).round(decimals=5))
        
    axs[0, 1].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[0, 1].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[0, 1].set_yticklabels('')
    axs[0, 1].set_ylabel('')
    axs[0, 1].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[0, 1].set_xticklabels(np.arange(0, 10.5*G_max/10, G_max/10).round(decimals=5))
    axs[0, 1].set_xlabel('$G$')    
    axs[0, 1].scatter(G_sp*(100/G_max), R_sp*(100/R_max), c=color, ec='w', s=100)    

    axs[1, 2].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[1, 2].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[1, 2].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[1, 2].set_xticklabels(np.arange(0, 10.5*x_s_max/10, x_s_max/10).round(decimals=0))

    axs[0, 2].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[0, 2].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[0, 2].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[0, 2].set_xticklabels(np.arange(0, 10.5*x_s_max/10, x_s_max/10).round(decimals=0))

    axs[1, 0].set_yticks(list(np.arange(9.5, 100.5, 10)))
    axs[1, 0].set_yticklabels(np.arange(R_max/10, 10.5*R_max/10, R_max/10).round(decimals=3))
    axs[1, 0].set_xticks(list(np.arange(0.5, 101.5, 10)))
    axs[1, 0].set_xticklabels(np.arange(0, 10.5*m_max/10, m_max/10).round(decimals=3))

    fig3.show()

    return


def plot_figure_1():

    A = 0.00001
    
    data0 = pd.DataFrame(columns=['M', 'G', 'R', 'D', 'age (years)', 'survival', 'species'])

    for M, G, R, x_s, hue in [[0.00001, 0.00041, 0.071, 13, 'Human'], [0.00001, 0.0094, 0.048, 11, 'Killer whale'], [0.05, 0.003, 0.2, 5, 'Yellow baboon'], 
                              [0.0522, 0.0025, 0.325, 2, 'Lion'], [0.106, 0.106, 0.0001, 1, 'Naked mole rat']]:
        D = round(float(death(R, G, M, A)),2)
        D_str = str(D)
        
        for x in np.arange(0, D, D/100):
            sur = survival(x, M, G, R)
            data0.loc[len(data0)] = [M, G, R, D, x, sur, hue]
        
    data1 = pd.DataFrame(columns=['M', 'G', 'senescence rate', 'D', 'R', 'x_s', 'Euler-Lotka selection', 'LRS selection', 'species'])

    for M, G, x_s, m, hue in [[0.00001, 0.00041, 13, 0.036, 'Human'], 
                              [0.00001, 0.0094, 11, 0.051, 'Killer whale'], 
                              [0.05, 0.003, 5, 0.182, 'Yellow baboon'], 
                              [0.0522, 0.0025, 2, 0.2, 'Lion']]:
        
        R_max = 1 
        for R in np.arange(0.001, R_max, R_max/300):
            D = round(float(death(R, G, M, A)),2)
            D_str = str(D)
            r = fitness(M, G, R, m, x_s, A)
            selection = MA2(R, M, G, r, m, x_s, A)
            LRS_selection = MA_LRS3(M, G, R, m, x_s, A)
            data1.loc[len(data1)] = [M, G, R, D_str, r, x_s, selection, LRS_selection, hue]


    data2 = pd.DataFrame(columns=['M', 'G', 'senescence rate', 'D', 'r', 'x_s', 'Euler-Lotka selection', 'LRS selection', 'species'])

    for M, G, x_s, m, hue in [[0.00001, 0.00041, 13, 0.036, 'Human'], 
                              [0.00001, 0.0094, 11, 0.051, 'Killer whale'], 
                              [0.05, 0.003, 5, 0.182, 'Yellow baboon'], 
                              [0.0522, 0.0025, 2, 0.2, 'Lion']]:
        
        
        R_max = 1 
        for R in np.arange(0.001, R_max, R_max/300):
            D = round(float(death(R, G, M, A)),2)
            D_str = str(D)
            M_dd = DD_external(M, G, R, m, x_s, A)
            r = fitness(M_dd, G, R, m, x_s, A)
            selection = MA2(R, M_dd, G, r, m, x_s, A)
            LRS_selection = MA_LRS(M, G, R, x_s, A)
            data2.loc[len(data2)] = [M, G, R, D_str, r, x_s, selection, LRS_selection, hue]

    data3 = pd.DataFrame(columns=['M', 'G', 'senescence rate', 'm', 'D', 'r', 'x_s', 'Euler-Lotka selection', 'species'])

    for M, G, x_s, m, hue in [[0.00001, 0.00041, 13, 0.036, 'Human'], 
                              [0.00001, 0.0094, 11, 0.051, 'Killer whale'], 
                              [0.05, 0.003, 5, 0.182, 'Yellow baboon'], 
                              [0.0522, 0.0025, 2, 0.2, 'Lion']]:
        
        R_max = 1 
        for k in [0.5, 2]:
            for R in np.arange(0.001, R_max, R_max/300):
                D = round(float(death(R, G, M, A)),2)
                D_str = str(D)
                m = DD_maternity(M, G, R, x_s, A)
                r = fitness(M, G, R, m*k, x_s, A)
                selection = MA2(R, M, G, r, m*k, x_s, A)
                data3.loc[len(data3)] = [M, G, R, k, D_str, r, x_s, selection, hue]

    fig, axs = plt.subplots(3, 2)

    axs[0, 1].text( -0.2, 0.2, '(B)', size=15)
    #sns.lineplot(data1[data1['species']!='Naked mole rat'], x='senescence rate', y='LRS selection', hue='species', palette=['darkred', 'darkblue', 'gold', 'darkorange', 'darkgrey'], ax=axs[0, 0]).set(ylim=(-50, 1))
    #axs[0, 0].text( -0.2, 2, '(A)', size=15)
    sns.lineplot(data1[data1['species']!='Naked mole rat'], x='senescence rate', y='Euler-Lotka selection', hue='species', palette=['darkred', 'darkblue', 'gold', 'darkorange', 'darkgrey'], ax=axs[0, 1], legend=False).set(ylim=(-5, 0.1))

    #axs[0, 0].set_title('density independent population')
    axs[0, 1].set_title('density independent population')

    #axs[0, 0].scatter(0.071, -25.9131, s=80, c='darkred', marker=(5, 2), zorder=20)
    axs[0, 1].scatter(0.071, -0.235052, s=80, c='darkred', marker=(5, 2), zorder=20)

    #axs[0, 0].scatter(0.0476667, -17.7623, s=80, c='darkblue', marker=(5, 2), zorder=20)
    axs[0, 1].scatter(0.0476667, -0.586506, s=80, c='darkblue', marker=(5, 2), zorder=20)

    #axs[0, 0].scatter(0.201, -4.50317, s=80, c='gold', marker=(5, 2), zorder=20)
    axs[0, 1].scatter(0.201, -0.251733, s=80, c='gold', marker=(5, 2), zorder=20)

    #axs[0, 0].scatter(0.334333, -2.90739, s=80, c='darkorange', marker=(5, 2), zorder=20)
    axs[0, 1].scatter(0.334333, -0.206484, s=80, c='darkorange', marker=(5, 2), zorder=20)


    axs[1, 1].text( -0.2, 0.2, '(D)', size=15)
    sns.lineplot(data2[data2['species']!='Naked mole rat'], x='senescence rate', y='LRS selection', hue='species', palette=['darkred', 'darkblue', 'gold', 'darkorange', 'darkgrey'], ax=axs[1, 0], legend=False).set(ylim=(-50, 1))
    axs[1, 0].text( -0.2, 2, '(C)', size=15)
    sns.lineplot(data2[data2['species']!='Naked mole rat'], x='senescence rate', y='Euler-Lotka selection', hue='species', palette=['darkred', 'darkblue', 'gold', 'darkorange', 'darkgrey'], ax=axs[1, 1], legend=False).set(ylim=(-5, 0.1))

    axs[1, 0].set_title('density dependence (reproduction)')
    axs[1, 1].set_title('density dependence (external risk)')

    axs[1, 0].scatter(0.071, -13.8512, s=80, c='darkred', marker=(5, 2), zorder=20)
    axs[1, 1].scatter(0.071, -0.234214, s=80, c='darkred', marker=(5, 2), zorder=20)

    axs[1, 0].scatter(0.0476667, -16.4377, s=80, c='darkblue', marker=(5, 2), zorder=20)
    axs[1, 1].scatter(0.0476667, -0.586142, s=80, c='darkblue', marker=(5, 2), zorder=20)

    axs[1, 0].scatter(0.201, -3.39307, s=80, c='gold', marker=(5, 2), zorder=20)
    axs[1, 1].scatter(0.201, -0.25085, s=80, c='gold', marker=(5, 2), zorder=20)

    axs[1, 0].scatter(0.334333, -1.96891, s=80, c='darkorange', marker=(5, 2), zorder=20)
    axs[1, 1].scatter(0.334333, -0.205587, s=80, c='darkorange', marker=(5, 2), zorder=20)


    axs[2, 0].text( -0.2, 0.2, '(E)', size=15)
    sns.lineplot(data3[(data3['m']==0.5) & (data3['species']!='Tundra vole')], x='senescence rate', y='Euler-Lotka selection', hue='species', palette=['darkred', 'darkblue', 'gold', 'darkorange', 'darkgrey'], ax=axs[2, 0], legend=False).set(ylim=(-5, 0.1))
    axs[2, 1].text( -0.2, 0.2, '(F)', size=15)
    sns.lineplot(data3[(data3['m']==2) & (data3['species']!='Tundra vole')], x='senescence rate', y='Euler-Lotka selection', hue='species', palette=['darkred', 'darkblue', 'gold', 'darkorange', 'darkgrey'], ax=axs[2, 1], legend=False).set(ylim=(-5, 0.1))

    axs[2, 0].set_title('declining population')
    axs[2, 1].set_title('growing population')

    axs[2, 0].scatter(0.071, -0.450347, s=80, c='darkred', marker=(5, 2), zorder=20)
    axs[2, 1].scatter(0.071, -0.225670, s=80, c='darkred', marker=(5, 2), zorder=20)

    axs[2, 0].scatter(0.0476667, -0.792253, s=80, c='darkblue', marker=(5, 2), zorder=20)
    axs[2, 1].scatter(0.0476667, -0.451073, s=80, c='darkblue', marker=(5, 2), zorder=20)

    axs[2, 0].scatter(0.201, -0.406634, s=80, c='gold', marker=(5, 2), zorder=20)
    axs[2, 1].scatter(0.201, -0.200422, s=80, c='gold', marker=(5, 2), zorder=20)

    axs[2, 0].scatter(0.334333, -0.392115, s=80, c='darkorange', marker=(5, 2), zorder=20)
    axs[2, 1].scatter(0.334333, -0.163711, s=80, c='darkorange', marker=(5, 2), zorder=20)


    axs[0, 0].text( -20, 1.1, '(A)', size=15)
    sns.lineplot(data0[data0['species']!='Naked mole rat'], x='age (years)', y='survival', ax=axs[0, 0], hue='species', palette=['darkred', 'darkblue', 'gold', 'darkorange', 'darkgrey']).set_title('Gompertz - Makeham survival')

    axs[0, 0].scatter(12, survival(13, 0.00001, 0.00041, 0.071), s=80, c='darkred', marker="*", zorder=20)    
    axs[0, 0].scatter(11, survival(11, 0.00001, 0.0094, 0.048), s=80, c='darkblue', marker="*", zorder=20)    
    axs[0, 0].scatter(5, survival(5, 0.05, 0.003, 0.2), s=80, c='gold', marker="*", zorder=20)    
    axs[0, 0].scatter(2, survival(2, 0.0522, 0.0025, 0.325), s=80, c='darkorange', marker="*", zorder=20)    
    #axs[0, 0].scatter(0.2, survival(0.5, 0.106, 0.106, 0.0000000001), s=80, c='darkgrey', marker=(5, 2), zorder=20)    

    fig.tight_layout(pad=0.2)
    
    plt.show()


def plot_figure_2():
    
    M_sp, G_sp, x_s_sp, m_sp, vmin_sp_EL, vmin_sp_LRS, c_sp, M_max, G_max = 0.05, 0.05, 10, 0.5, -2, -200, 'darkgrey', 0.1, 0.1

    A = 0.0001
    res = 100
    
    dRdr4 = pd.DataFrame(columns=['$G$', '$ζ$', 'Euler Lotka selection', 'LRS selection'])

    for i in range(0, res+1):
        G = G_max*i/res
        G = round(G, 6)
        for j in range(1, res+1):
            m = j/res
            m = round(m, 6)
            print(m)
            if neglectable_death(G, M_sp, A)>x_s_sp:
                
                dRdr = neglectable_EL(G, x_s_sp, m) 
                dRdr_LRS = neglectable_LRS(M_sp, G, x_s_sp)
                if G>0:
                    if x_s_sp>1/(M_sp*G):
                        print('!!!')
            else:
                dRdr = np.nan
                dRdr_LRS = np.nan
            
            dRdr4.loc[len(dRdr4)] = {'$G$':G, '$ζ$':m, 'Euler Lotka selection':dRdr, 'LRS selection':dRdr_LRS}

    dRdr5 = pd.DataFrame(columns=['$x_s$', '$ζ$', 'Euler Lotka selection', 'LRS selection'])

    for i in range(1, res+1):
        m = i/res
        m = round(m, 6)
        for j in range(1, res+1):
            x_s = 20*j/res
            x_s = round(x_s, 4)
            if neglectable_death(G_sp, M_sp, A)>x_s:
                
                dRdr = neglectable_EL(G_sp, x_s, m) 
                dRdr_LRS = neglectable_LRS(M_sp, G_sp, x_s)
                if G_sp>0:
                    if x_s>1/(M_sp*G_sp):
                        print('!!!')
            else:
                dRdr = np.nan
                dRdr_LRS = np.nan
            
            dRdr5.loc[len(dRdr5)] = {'$x_s$':x_s, '$ζ$':m, 'Euler Lotka selection':dRdr, 'LRS selection':dRdr_LRS}

    dRdr6 = pd.DataFrame(columns=['$M$', '$x_s$', 'Euler Lotka selection', 'LRS selection'])

    for i in range(1, res+1):
        M = M_max*i/res
        M = round(M, 6)
        for j in range(1, res+1):
            x_s = 20*j/res
            x_s = round(x_s, 4)
            if neglectable_death(G_sp, M, A)>x_s:
                
                dRdr = neglectable_EL(G_sp, x_s, m_sp) 
                dRdr_LRS = neglectable_LRS(M, G_sp, x_s)
                if G_sp>0:
                    if x_s>1/(M*G_sp):
                        print('!!!')
            else:
                dRdr = np.nan
                dRdr_LRS = np.nan
            
            dRdr6.loc[len(dRdr6)] = {'$M$':M, '$x_s$':x_s, 'Euler Lotka selection':dRdr, 'LRS selection':dRdr_LRS}
            
    dRdr0 = pd.DataFrame(columns=['$G$', '$x_s$', 'Euler Lotka selection', 'LRS selection'])

    for i in range(0, res+1):
        G = G_max*i/res
        G = round(G, 6)
        for j in range(1, res+1):
            x_s = 20*j/res
            x_s = round(x_s, 3)
            if neglectable_death(G, M_sp, A)>x_s:
                
                dRdr = neglectable_EL(G, x_s, m_sp) 
                dRdr_LRS = neglectable_LRS(M_sp, G, x_s)
                if G>0:
                    if x_s>1/(M_sp*G):
                        print('!!!')
            else:
                dRdr = np.nan
                dRdr_LRS = np.nan
            
            dRdr0.loc[len(dRdr0)] = {'$G$':G, '$x_s$':x_s, 'Euler Lotka selection':dRdr, 'LRS selection':dRdr_LRS}
            
    dRdr3 = pd.DataFrame(columns=['$G$', '$M$', 'LRS selection'])

    for i in range(0, res+1):
        G = G_max*i/res
        G = np.longdouble(round(G, 6))
        for j in range(1, res+1):
            M = M_max*j/res
            M = np.longdouble(round(M, 6))
            if neglectable_death(G, M, A)>x_s_sp:
                dRdr_LRS = neglectable_LRS(M, G, x_s_sp)
                print(neglectable_LRS(M, G, x_s_sp))
            else:
                dRdr_LRS = np.nan
            dRdr3.loc[len(dRdr3)] = {'$G$':G, '$M$':M, 'LRS selection':dRdr_LRS}
            
    fig, axs = plt.subplots(2, 3, figsize=(10,5))

    prep_dRdr0 = dRdr0.pivot(index='$G$', columns='$x_s$', values='Euler Lotka selection')
    sns.heatmap(prep_dRdr0, cbar_kws={'label': 'Euler Lotka selection'}, cmap='magma', ax=axs[0, 0], square=True, vmin=vmin_sp_EL).invert_yaxis()
    axs[0, 0].text(-10, 110, '(A)', size=15)
    axs[0, 0].set_xticks(list(np.arange(9.5, 100.5, 10)))
    axs[0, 0].set_xticklabels(list(range(2, 21, 2)))
    axs[0, 0].set_yticks(list(np.arange(0.5, 101.5, 10)))
    axs[0, 0].set_yticklabels(np.arange(0, 10.5*G_max/10, G_max/10).round(len(str(G_max).split('.')[1])+1))

    axs[0, 0].scatter(x_s_sp*5, G_sp*100/G_max, s=80, c=c_sp, marker="o", zorder=20) 

    prep_dRdr4 = dRdr4.pivot(index='$G$', columns='$ζ$', values='Euler Lotka selection')        
    sns.heatmap(prep_dRdr4, cbar_kws={'label': 'Euler Lotka selection'}, cmap='magma', ax=axs[0, 1], square=True, vmin=vmin_sp_EL).invert_yaxis()
    axs[0, 1].text(-10, 110, '(B)', size=15)
    axs[0, 1].set_xticks(list(np.arange(9.5, 100.5, 10)))
    axs[0, 1].set_xticklabels(np.array(list(range(1, 11, 1)))/10)
    axs[0, 1].set_yticks(list(np.arange(0.5, 101.5, 10)))
    axs[0, 1].set_yticklabels(np.arange(0, 10.5*G_max/10, G_max/10).round(len(str(G_max).split('.')[1])+1))

    axs[0, 1].scatter(m_sp*100, G_sp*100/G_max, s=80, c=c_sp, marker="o", zorder=20) 

    prep_dRdr5 = dRdr5.pivot(index='$ζ$', columns='$x_s$', values='Euler Lotka selection')        
    sns.heatmap(prep_dRdr5, cbar_kws={'label': 'Euler Lotka selection'}, cmap='magma', ax=axs[0, 2], square=True, vmin=vmin_sp_EL).invert_yaxis()
    axs[0, 2].text(-10, 110, '(C)', size=15)
    axs[0, 2].set_xticks(list(np.arange(9.5, 100.5, 10)))
    axs[0, 2].set_xticklabels(np.array(list(range(2, 21, 2))))
    axs[0, 2].set_yticks(list(np.arange(0.5, 101.5, 10)))
    axs[0, 2].set_yticklabels(np.array(list(range(0, 11, 1)))/10)

    axs[0, 2].scatter(x_s_sp*5, m_sp*100, s=80, c=c_sp, marker="o", zorder=20) 

    prep_dRdr0 = dRdr0.pivot(index='$G$', columns='$x_s$', values='LRS selection')        
    sns.heatmap(prep_dRdr0, cbar_kws={'label': 'LRS selection'}, cmap='magma', ax=axs[1, 0], square=True, vmin=vmin_sp_LRS).invert_yaxis()
    axs[1, 0].text(-10, 110, '(D)', size=15)
    axs[1, 0].set_xticks(list(np.arange(9.5, 100.5, 10)))
    axs[1, 0].set_xticklabels(list(range(2, 21, 2)))
    axs[1, 0].set_yticks(list(np.arange(0.5, 101.5, 10)))
    axs[1, 0].set_yticklabels(np.arange(0, 10.5*G_max/10, G_max/10).round(len(str(G_max).split('.')[1])+1))

    axs[1, 0].scatter(x_s_sp*5, G_sp*100/G_max, s=80, c=c_sp, marker="o", zorder=20) 

    prep_dRdr3 = dRdr3.pivot(index='$G$', columns='$M$', values='LRS selection')        
    sns.heatmap(prep_dRdr3, cbar_kws={'label': 'LRS selection'}, cmap='magma', ax=axs[1, 1], square=True, vmin=vmin_sp_LRS).invert_yaxis()
    axs[1, 1].text(-10, 110, '(E)', size=15)
    axs[1, 1].set_xticks(list(np.arange(9.5, 100.5, 10)))
    axs[1, 1].set_xticklabels(np.arange(M_max/10, 10.5*M_max/10, M_max/10).round(len(str(M_max).split('.')[1])+1))
    axs[1, 1].set_yticks(list(np.arange(0.5, 101.5, 10)))
    axs[1, 1].set_yticklabels(np.arange(0, 10.5*G_max/10, G_max/10).round(len(str(G_max).split('.')[1])+1))

    axs[1, 1].scatter(M_sp*100/M_max, G_sp*100/G_max, s=80, c=c_sp, marker="o", zorder=20) 

    prep_dRdr6 = dRdr6.pivot(index='$M$', columns='$x_s$', values='LRS selection')        
    sns.heatmap(prep_dRdr6, cbar_kws={'label': 'LRS selection'}, cmap='magma', ax=axs[1, 2], square=True, vmin=vmin_sp_LRS).invert_yaxis()
    axs[1, 2].text(-10, 110, '(F)', size=15)
    axs[1, 2].set_xticks(list(np.arange(9.5, 100.5, 10)))
    axs[1, 2].set_xticklabels(np.array(list(range(2, 21, 2))))
    axs[1, 2].set_yticks(list(np.arange(9.5, 101.5, 10)))
    axs[1, 2].set_yticklabels(np.arange(M_max/10, 10.5*M_max/10, M_max/10).round(len(str(M_max).split('.')[1])+1))

    axs[1, 2].scatter(x_s_sp*5, M_sp*100/M_max, s=80, c=c_sp, marker="o", zorder=20) 

    fig.tight_layout(pad=0.1)
    fig.show()
    return




#%% styles

plt.style.use('grayscale')
plt.style.use('bmh')

#%% generate plots

# insert function to plot:

plot_figure_1()

plot_figure_2()

# select species between 'human', 'baboon', 'lion', and 'whale'
plot_heatmaps_for_figures_3_and_4(specie='baboon')

plot_figure_5()





