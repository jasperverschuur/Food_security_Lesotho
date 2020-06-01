#!/usr/bin/env python

import scipy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.stats as stats
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor,LinearRegression


from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import norm

from sklearn import linear_model
from scipy.stats import genextreme as gev

import netCDF4 as nc
from netCDF4 import Dataset


import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf

import random

plt.rcParams["font.family"] = "Times New Roman"

data = pd.read_csv('agricultural_data.csv') ### for lag index = 26 instead of 28
L_rain = pd.read_csv('Lesotho-ERA5.csv')
SA_rain = pd.read_csv('SA-ERA5.csv')
price_data = pd.read_excel('price_SA.xlsx')[2:]

index_2007 = 26

#SA the values are respectively 2.15 (95% CI: 1.63 – 2.67) and 1.78 (range: 0.5 – 9.16) CMIP5.
m_nat_tot = []
m_nat_nt_tot = []
m_act_nt_tot = []

shor_ano_a = []
shor_ano_n = []
shor_ano_nt = []

a_nat_tot = []
a_nat_nt_tot = []
a_act_nt_tot = []
a_act_tot = []

nat_frac = []
nat_nt_frac = []
act_nt_frac = []

rain_L = []
rain_SA = []

price_a = []
price_n = []
price_n_nt = []

price_a_abs = []
price_n_abs = []


N = 100

RR_L_sampler = np.random.triangular(1.51,5.36, 32.5, N)
RR_SA_sampler = np.random.triangular(1.53, 4.70, 26.3, N)

RR_L_df = pd.DataFrame(RR_L_sampler).to_csv('output_L_SA/RR_L.csv',index = False)
RR_SA_df = pd.DataFrame(RR_SA_sampler).to_csv('output_L_SA/RR_SA.csv',index = False)

for i in range(0,N):

    RR_L = RR_L_sampler[i]
    RR_SA = RR_SA_sampler[i]
    print(i,RR_L,RR_SA)
    L_return = 40/RR_L

    SA_return = 40/RR_SA

    #L_rain = pd.read_csv('/Users/Jasper/Lesotho-ERA5.csv')
    #SA_rain = pd.read_csv('/Users/Jasper/SA-ERA5.csv')


    return_period = np.linspace(1,len(L_rain),len(L_rain))
    return_period = return_period / (len(return_period)+1)
    L_rain = L_rain.sort_values(by=['JFM_prec'])
    SA_rain = SA_rain.sort_values(by=['JFM_prec'])


    shape_SA, loc_SA, scale_SA = gev.fit(SA_rain['JFM_prec'])
    xx_SA = np.linspace(100, 1000, 1000)
    yy_SA = 1/(gev.cdf(xx_SA, shape_SA, loc_SA, scale_SA))


    shape_L, loc_L, scale_L = gev.fit(L_rain['JFM_prec'])
    xx_L = np.linspace(100, 1000, 1000)
    yy_L = 1/(gev.cdf(xx_L, shape_L, loc_L, scale_L))


    ### find the index
    id_SA_return1 = (np.abs(yy_SA-SA_return)).argmin()
    val_SA_return = xx_SA[id_SA_return1]


    id_L_return1 = (np.abs(yy_L-L_return)).argmin()
    val_L_return= xx_L[id_L_return1]


    ### find the index
    id_SA_return2 = (np.abs(yy_SA-40)).argmin()
    val_SA_return_ACT = xx_SA[id_SA_return2]


    id_L_return2 = (np.abs(yy_L-40)).argmin()
    val_L_return_ACT= xx_L[id_L_return2]


    production_SA_detrend = signal.detrend(data['production-SA'])
    rain_SA_detrend = signal.detrend(data['rain-SA'])
    rain_SA_lag1_detrend = signal.detrend(data['rain-SA-lag1'])
    rain_SA_lag2_detrend = signal.detrend(data['rain-SA-lag2'])

    rain_L_detrend = signal.detrend(data['rain-L'])
    rain_L_lag1_detrend = signal.detrend(data['rain-L-lag1'])
    rain_L_lag2_detrend = signal.detrend(data['rain-L-lag2'])

    #### fit a lowess with frac 1/2 to the shortage data

    x1 = np.arange(0,len(data['Shortage']))
    lowess = sm.nonparametric.lowess
    z_detrend = lowess(data['Shortage'],x1,frac = 1/1)

    shortage_detrend = data['Shortage']-z_detrend[:, 1]

    #### check if trend in production is significant
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['year'],data['Shortage'])
    slope1, intercept1,r_value1, p_value1, std_err1 = stats.linregress(data['year'].iloc[15:],data['Shortage'].iloc[15:])


    ### find the detrend line of rain SA and rain L
    #print(data['rain-SA'].iloc[index_2007])
    data1 = data['rain-SA']
    x = np.arange(0,len(data1))
    y=np.array(data1)
    z = np.polyfit(x,y,1)
    #
    val_rain_SA = data1[index_2007] - 314.6555335957142 + 0.47705378614285701*index_2007


    data2 = data['rain-L']
    x = np.arange(0,len(data2))
    y=np.array(data2)
    z = np.polyfit(x,y,1)

    val_rain_L = data2[index_2007] - 455.47557053939386 - 0.1802992386363648*index_2007

    ### get the detrended values of NAT rain 2007

    def detrend_SA(value):
        val = value - 314.6555335957142 + 0.47705378614285701*index_2007
        return val

    def detrend_L(value):
        val = value - 455.47557053939386 - 0.1802992386363648*index_2007
        return val

    val_SA_return_det = detrend_SA(val_SA_return)
    val_L_return_det = detrend_L(val_L_return)
    rain_L.append(val_L_return)
    rain_SA.append(val_SA_return)

    val_SA_return_ACT_det = detrend_SA(val_SA_return_ACT)
    val_L_return_ACT_det = detrend_L(val_L_return_ACT)

    ## get the value to add it up
    data5 = data['production-SA']
    x = np.arange(0,len(data5))
    y=np.array(data5)
    z_prod = np.polyfit(x,y,1)


    def retrend_SA_prod(value):
        val = value +7823.836007130121 + 97.02540106951885*index_2007
        return val


    ## get the value to add it up
    data4 = data['Shortage']
    x = np.arange(0,len(data4))
    y=np.array(data4)
    z_shortage = np.polyfit(x,y,1)
    "{0}x + {1}".format(*z_shortage)


    def retrend_L_shortage(value):
        val = value +54.44935351158645 + 3.601995026737967*index_2007
        return val


    val_rain_NA_SA_detrend = val_SA_return_det
    val_rain_ACT_SA_detrend = val_SA_return_ACT_det

    dummy = np.where(rain_SA_detrend<0,1,0)
    dummy1 = np.where(rain_SA_lag1_detrend<0,1,0)
    dummy2 = np.where(rain_SA_lag2_detrend<0,1,0)

    ### ypoly fit
    x_part = data.filter(['rain-SA','rain-SA-lag1','rain-SA-lag2','production-SA'], axis=1)
    x_part['rain-SA'] = rain_SA_detrend
    x_part['rain-SA-lag1'] = rain_SA_lag1_detrend
    x_part['rain-SA-lag2'] = rain_SA_lag2_detrend
    x_part['dummy'] = dummy
    x_part['dummy1'] = dummy1
    x_part['production-SA'] = production_SA_detrend

    x_part = x_part[['rain-SA', 'rain-SA-lag1','rain-SA-lag2','dummy','dummy1','production-SA']]


    X = x_part.iloc[:, 0:5].values
    y = x_part.iloc[:, 5].values

    #### get the polynomial fit
    poly = PolynomialFeatures(degree = 2)
    X_poly = poly.fit_transform(X)

    #poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)

    prod_pred = lin2.predict(poly.fit_transform(X))
    prod_NAT = lin2.predict(poly.fit_transform([[val_rain_NA_SA_detrend,rain_SA_lag1_detrend[index_2007], rain_SA_lag2_detrend[index_2007],1,0]]))
    prod_ACT = lin2.predict(poly.fit_transform([[val_rain_ACT_SA_detrend,rain_SA_lag1_detrend[index_2007],rain_SA_lag2_detrend[index_2007],1,0]]))


    ##### Lesotho
    dummy = np.where(rain_L_detrend<0,1,0)
    dummy1 = np.where(rain_L_lag1_detrend<0,1,0)
    val_rain_NA_L_detrend = val_L_return_det
    val_rain_ACT_L_detrend = val_L_return_ACT_det


    ### ypoly fit
    x_part = data.filter(['rain-L','rain-L-lag1','rain-L-lag2','Shortage'], axis=1)
    x_part['rain-L'] = rain_L_detrend
    x_part['rain-L-lag1'] = rain_L_lag1_detrend
    x_part['rain-L-lag2'] = rain_L_lag2_detrend
    x_part['dummy'] = dummy
    x_part['dummy1'] = dummy1
    x_part['Shortage'] = shortage_detrend

    x_part = x_part[['rain-L', 'rain-L-lag1','rain-L-lag2','dummy','dummy1','Shortage']]

    X = x_part.iloc[:, 0:5].values
    y = x_part.iloc[:, 5].values

    #print(X[28])
    #### get the polynomial fit
    poly = PolynomialFeatures(degree = 2)
    X_poly = poly.fit_transform(X)


    #poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)
    shortage_pred = lin2.predict(poly.fit_transform(X))
    short_NAT = lin2.predict(poly.fit_transform([[val_rain_NA_L_detrend, rain_L_lag1_detrend[index_2007],rain_L_lag2_detrend[index_2007],1,0]]))
    short_ACT = lin2.predict(poly.fit_transform([[val_rain_ACT_L_detrend,rain_L_lag1_detrend[index_2007],rain_L_lag2_detrend[index_2007],1,0]]))

    #### now get the price of maize and rainfall South Africa

    price_data = pd.read_excel('price_SA.xlsx')[2:]
    x1 = np.arange(0,len(price_data['Value']))
    lowess = sm.nonparametric.lowess
    z_price = lowess(price_data['Value'],x1,frac = 1/2)

    price_data['price_detrend']= price_data['Value']-z_price[:, 1]

    price_data['rain-SA-d'] = signal.detrend(price_data['rain-SA'])
    price_data['rain-SA-d-lag1'] = signal.detrend(price_data['rain-SA-lag1'])
    price_data['rain-SA-d-lag2'] = signal.detrend(price_data['rain-SA-lag2'])
    price_data['dummy']  = np.where(price_data['rain-SA-d']<0,1,0)
    price_data['dummy1']  = np.where(price_data['rain-SA-d-lag1']<0,1,0)


    y = price_data['price_detrend'].values
    X = price_data[['rain-SA-d','rain-SA-d-lag1','rain-SA-d-lag2','dummy','dummy1']]

    poly = PolynomialFeatures(degree = 2)
    X_poly = poly.fit_transform(X)

    lin_poly = LinearRegression()
    lin_poly.fit(X_poly, y)
    price_SA = lin_poly.predict(poly.fit_transform(X))

    diff_rain = val_rain_ACT_SA_detrend-val_rain_NA_SA_detrend
    price_NAT = lin_poly.predict(poly.fit_transform([[price_data['rain-SA-d'][16]-diff_rain,price_data['rain-SA-d-lag1'][16],price_data['rain-SA-d-lag2'][16],1,0]]))
    price_ACT = lin_poly.predict(poly.fit_transform([[price_data['rain-SA-d'][16],price_data['rain-SA-d-lag1'][16],price_data['rain-SA-d-lag2'][16],1,0]]))
    price_data['prediction']= price_SA


    ### now get the
    #### calculate the error
    err_L = shortage_detrend- shortage_pred #shortage_detrend_sort - shortage_pred
    err_SA1 = production_SA_detrend - prod_pred
    err_price = price_data['price_detrend'] -price_data['prediction']

    std_L = np.std(err_L)
    std_SA1 = np.std(err_SA1)

    #### get a distribution of potential values for shortage and export for 2007 event

    N_samples = 1


    error_L = norm.rvs(0, std_L, size=N_samples)
    error_SA_P = norm.rvs(0, std_SA1, size=N_samples)

    error_L_NA = norm.rvs(0, std_L, size=N_samples)
    error_L_NA_nt = norm.rvs(0, std_L, size=N_samples)
    error_SA_P_NA = norm.rvs(0, std_SA1, size=N_samples)

    error_L_ACT = norm.rvs(0, std_L, size=N_samples)
    error_SA_P_ACT = norm.rvs(0, std_SA1, size=N_samples)

    error_price_NAT = norm.rvs(0, np.std(err_price), size=N_samples)
    error_price_NAT_nt = norm.rvs(0, np.std(err_price), size=N_samples)
    error_price_ACT = norm.rvs(0, np.std(err_price), size=N_samples)

    #### export values
    val_export_act =2.0
    val_export_nat = np.random.uniform(0.5,2.5,N_samples)
    val_export_nat1 = np.random.uniform(0.5,2.5,N_samples)

    ### export SA

    pred_NA_shortage_nt = short_NAT + error_L_NA_nt

    pred_NA_shortage = short_NAT + error_L_NA
    pred_NA_prod = prod_NAT + error_SA_P_NA #### index 27 means 2007


    pred_ACT_shortage = short_ACT + error_L_ACT
    pred_ACT_prod = prod_ACT + error_SA_P_ACT


    pred_NA_prod = retrend_SA_prod(pred_NA_prod)
    pred_ACT_prod = retrend_SA_prod(pred_ACT_prod)


    shor_ano_a.append(pred_ACT_shortage[0])
    shor_ano_n.append(pred_NA_shortage[0])
    shor_ano_nt.append(pred_NA_shortage_nt[0])

    pred_NA_shortage = pred_NA_shortage +z_detrend[index_2007][1]
    pred_NA_shortage_nt = pred_NA_shortage_nt +z_detrend[index_2007][1]
    pred_ACT_shortage = pred_ACT_shortage+z_detrend[index_2007][1]

    pred_price_NAT = price_NAT+error_price_NAT
    pred_price_ACT = price_ACT+error_price_ACT
    pred_price_NAT_nt = price_NAT+error_price_NAT_nt

    pred_price_NAT =pred_price_NAT + z_price[14][1]
    pred_price_NAT_nt =pred_price_NAT_nt + z_price[14][1]
    pred_price_ACT =pred_price_ACT + z_price[14][1]

    ### calculate the food security

    secur_ind_NAT = pred_NA_prod*(val_export_nat/100)-pred_NA_shortage
    secur_ind_ACT_GEV = pred_ACT_prod*(val_export_act/100)-pred_ACT_shortage
    secur_ind_NAT_nt = pred_NA_prod*(val_export_nat1 /100)-pred_NA_shortage_nt+116
    print(secur_ind_ACT_GEV,secur_ind_NAT,secur_ind_NAT_nt)
    ###price of imports
    total_price_import_NAT = pred_price_NAT*pred_NA_shortage*1000/(1e6)
    total_price_import_NAT_nt = pred_price_NAT_nt*(pred_NA_shortage_nt-116)*1000/(1e6)
    total_price_import_ACT = pred_price_ACT*pred_ACT_shortage*1000/(1e6)

    diff_mean_nat = secur_ind_NAT- secur_ind_ACT_GEV
    diff_mean_nat_nt = secur_ind_NAT_nt- secur_ind_ACT_GEV


    abs_mean_nat = secur_ind_NAT
    abs_mean_nat_nt = secur_ind_NAT_nt
    abs_mean_act = secur_ind_ACT_GEV

    m_nat_tot.append(diff_mean_nat[0])
    m_nat_nt_tot.append(diff_mean_nat_nt[0])

    a_nat_tot.append(abs_mean_nat[0])
    a_nat_nt_tot.append(abs_mean_nat_nt[0])
    a_act_tot.append(abs_mean_act[0])

    nat_frac.append(val_export_nat[0])
    nat_nt_frac.append(val_export_nat1[0])

    price_a.append(total_price_import_ACT[0])
    price_n.append(total_price_import_NAT[0])
    price_n_nt.append(total_price_import_NAT_nt[0])

    price_a_abs.append(pred_price_ACT[0])
    price_n_abs.append(pred_price_NAT[0])


data_nat = pd.DataFrame({'anomaly':shor_ano_n[:],'diff_mean':m_nat_tot[:],'abs_mean':a_nat_tot[:],'price_abs':price_n_abs[:],'price':price_n[:],'frac':nat_frac[:],'rain_L':rain_L,'rain_SA':rain_SA}).to_csv('output_L_SA/data_nat.csv',index = False)
data_nat_nt = pd.DataFrame({'anomaly':shor_ano_nt[:],'diff_mean':m_nat_nt_tot[:],'abs_mean':a_nat_nt_tot[:],'price_abs':price_n_abs[:],'price':price_n_nt[:],'frac':nat_nt_frac[:],'rain_L':rain_L,'rain_SA':rain_SA}).to_csv('output_L_SA/data_nat_nt.csv',index = False)
data_act= pd.DataFrame({'anomaly':shor_ano_a[:],'abs_mean':a_act_tot[:],'price_abs':price_a_abs[:],'price':price_a[:],'frac':val_export_act,'rain_L':rain_L,'rain_SA':rain_SA}).to_csv('output_L_SA/data_act.csv',index = False)

#plt.gca().spines['left'].set_color('none')
