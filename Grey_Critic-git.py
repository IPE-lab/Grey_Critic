#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy.stats import bartlett

def Critic(df0):
    df0_var = list(df0.std())[:-1] 

    def conflict(X):  
        X_corr = X.corr()  
        l = []
        for i in range(len(X_corr)-1): 
            l.append(len(X_corr) - X_corr.iloc[i].sum())
        return l

    df0_con = conflict(df0)
    weight = []
    for i in range(len(df0_var)):
        weight.append(df0_var[i] * df0_con[i])
    weight = [round(i/sum(weight), 2) for i in weight]
    ref = df0.iloc[:, -1]
    com = df0.iloc[:, :-1]
    m, n = com.shape[0], com.shape[1]
    a = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            a[i,j] = abs(com.iloc[i,j] - ref[j])

    a_max, a_min = a.max(), a.min() 

    coe = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            coe[i, j] = (a_min + 0.5 * a_max) / (a[i, j] + 0.5 * a_max)
    coe_list = [coe[:, i].mean() for i in range(n)]
    c_coe_list = [coe_list[i] * weight[i] for i in range(n)]
    c_weight = [round(i/sum(c_coe_list), 2) for i in c_coe_list]
    return c_weight

def Grey_Critic(Geofilepath, Engfilepath):
    df1 = pd.read_excel(Geofilepath, header=0)
    df2 = pd.read_excel(Engfilepath, header=0)
    df01=df1.copy()
    df02=df2.copy()
    weight_1 = Critic(df01)
    weight_2 = Critic(df02)
    gw = []
    ew = []
    for i in range(len(df01)):
        gw.append(df01.iloc[i, 0]*weight_1[0] + df01.iloc[i, 1]*weight_1[1] + df01.iloc[i, 2]*weight_1[2] + df01.iloc[i,3]*weight_1[3])    
    for i in range(len(df01)):
        ew.append(df02.iloc[i, 0]*weight_2[0] + df02.iloc[i, 1]*weight_2[1] + df02.iloc[i, 2]*weight_2[2] + df02.iloc[i, 3]*weight_2[3] + df02.iloc[i, 4]*weight_2[4] + df02.iloc[i, 5]*weight_2[5])
    df_gc = pd.DataFrame({'Geology Sweetness Index':gw, 'Engineering Sweetness Index':ew, 'production_normalization':df01.iloc[:, -1]})
    return df_gc

