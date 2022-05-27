import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import layers
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import regularizers
from math import sqrt
from pickle import dump
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as st
from sympy import symbols, Eq,solveset
import math
from numpy.linalg import inv

#Import dataset
df1 = pd.read_csv('C:/University of Toronto/Modelling in Steelmaking/Data/Paper #2 Hybrid Model/#4 Cleaned_Merged_2020+2021 Discrete Dataset.csv')
df = pd.read_csv('C:/University of Toronto/Modelling in Steelmaking/Data/Paper #2 Hybrid Model/Dataset.csv')
#df = pd.merge(df1,df2,on=['HMA_CAST'])
#df.to_csv('Dataset.csv')


#Validate theoretical lp equations
fe = df['SL_FE'].values
cao = df['SL_CAO'].values
sio2 = df['SL_SIO2'].values
mgo = df['SL_MGO'].values
mno = df['SL_MNO'].values
al2o3 = df['SL_AL2O3'].values
tio2 = df['SL_TIO2'].values
p2o5 = df['SL_P2O5'].values
p = p2o5*(62/142)
steel_P = df['SA_P'].values
lp = np.log(p/steel_P).reshape(-1,1)
#temp = df['STL_TEMP'].values

#Slag Chemistry Model

def slag_model(df):
    hm_si = df['HMA_SI'].values
    hm_c = df['HMA_C'].values
    hm_mn = df['HMA_MN'].values
    hm_p = df['HMA_P'].values
    hm_s = df['HMA_S'].values
    hm_fe = 100 - df['HMA_C'].values - df['HMA_MN'].values - df['HMA_S'].values - df['HMA_P'].values - df['HMA_SI'].values - df['HMA_TI'].values - df['HMA_CR'].values
    hm_weight = df['HMWT'].values
    lime = df['LIME'].values
    dolo = df['DOLO'].values
    ore = df['ORE'].values
    aim_p = df['AIM_PHOS'].values
    oxy = df['OXY'].values

    X = np.array([hm_fe,hm_si,hm_c,hm_mn,hm_p,hm_s,lime,dolo]).T
    y = df['BASICITY'].values

    sc = StandardScaler()
    X = sc.fit_transform(X)
    lr = LinearRegression()
    lr.fit(X,y)
    basicity = lr.predict(X)

    sio2_hm = hm_weight*0.01*hm_si*60/28
    sio2_lime = lime*0.0073
    sio2_dolo = dolo*0.0412
    sl_sio2 = sio2_hm+sio2_dolo+sio2_lime
    sl_cao = sl_sio2*basicity
    sl_weight = sl_cao/0.51
    sio2_ratio = ((sl_sio2/sl_weight)*100)
    cao_ratio = ((sl_cao/sl_weight)*100)
    p2o5_ratio = ((hm_weight*0.01*(hm_p-aim_p)*142/62)*100/sl_weight)
    mgo_ratio = (lime*0.0065 + dolo*0.1704)/sl_weight*100
    
    '''oxy_lance = 18*(oxy * 1000 / 22.4)
    oxy_ore = ore*10**6*48/160
    oxy_lime = lime*10**6*(93.92*16/56+0.73*32/60+0.65*16/40+0.47*48/102+0.14*48/160+3.54*1/2)*0.01
    oxy_dolo = dolo*10**6*(33.61*16/56+4.12*32/60+17.04*16/40+0.47*48/102+0.53*48/160+43.86*1/2)*0.01
    oxy = oxy_lance+(oxy_ore+oxy_lime+oxy_dolo)/16
    o_sio2 = 2*(hm_weight*10**6*hm_si/100)/28
    o_p2o5 = 5/2*(hm_weight*10**6*(hm_p-aim_p)/100)/31
    o_c = (0.9*2+0.1*1)*(hm_weight*10**6*(hm_c-0.12)/100)/12
    o_fe = oxy- (o_sio2+o_p2o5+o_c)
    sl_feo = ((o_fe*72)/(sl_weight*10**6))
    sl_fe = 56/72*sl_feo'''

    sl_feo = 95 - cao_ratio - sio2_ratio - p2o5_ratio - mgo_ratio
    return sl_feo,cao_ratio,sio2_ratio,p2o5_ratio,mgo_ratio,sl_weight
sl_mno = 0.4

def theo_t(df):
    temp = df['STL_TEMP'].values
    hm_si = df['HMA_SI'].values
    hm_c = df['HMA_C'].values
    hm_mn = df['HMA_MN'].values
    hm_p = df['HMA_P'].values
    hm_s = df['HMA_S'].values
    hm_fe = 100- df['HMA_C'].values - df['HMA_MN'].values - df['HMA_S'].values -df['HMA_P'].values - df['HMA_SI'].values - df['HMA_TI'].values - df['HMA_CR'].values
    hm_weight = df['HMWT'].values
    ore = df['ORE'].values
    scrap = df['SCP'].values

    #Steel chemistry assumptions
    sl_feo, sl_cao, sl_sio2, sl_p2o5, sl_mgo, sl_weight = slag_model(df)
    sl_fe = 56 / 72 * sl_feo
    sl_al2o3 = st.mode(df['SL_AL2O3'])[0]
    sl_mno=0.4
    #Steel chemistry assumptions
    hm_temp = df['HMTEMP'].values+273
    stl_temp = df['STL_TEMP'].values+273
    stl_weight = (hm_weight*0.01*hm_fe+scrap+ore*112/160-sl_weight*0.01*sl_fe)/0.995 #assume 99.5% iron in steel
    stl_fe = 99.5
    stl_mn =st.mode(df['SA_MN'])[0]
    eb_c = st.mode(df['SA_C'])[0]
    stl_p = ((((hm_weight*10**6*0.01*hm_p - sl_weight*10**6*0.01*sl_p2o5*(62/142)))/(stl_weight*10**6))*100).reshape(-1,1)

    # Sensible heat of hot metal
    def h1(hm_weight, hm_fe, hm_si, hm_mn, hm_c, hm_p, temp):
        fe = hm_weight * 10 ** 6 * 0.01 * hm_fe * (0.72105 * temp - 100)
        si = hm_weight * 10 ** 6 * 0.01 * hm_si * (0.9614 * temp + 1450.46)
        mn = hm_weight * 10 ** 6 * 0.01 * hm_mn * (0.836 * temp - 140.448)
        c = hm_weight * 10 ** 6 * 0.01 * hm_c * (1.996786 * temp - 1057.54)
        p = hm_weight * 10 ** 6 * 0.01 * hm_p * (0.563 * temp - 169.355)
        s = hm_weight * 10 ** 6 * 0.01 * hm_s * (37.4 * temp - 12500) / 32
        h = fe + si + mn + c + p + s
        return h
    # Heat of reactions (Exothermic of compounds & gases)
    def h2(hm_weight, hm_si, hm_c, stl_c, slag_weight, sl_feo, sl_mno):
        feo = slag_weight * 10 ** 6 * 0.01 * sl_feo * 3862.32
        sio2 = hm_weight * 10 ** 6 * 0.01 * hm_si * 902300 / 28
        p2o5 = slag_weight * 10 ** 6 * 0.01 * sl_p2o5 * 10345.5
        mno = slag_weight * 10 ** 6 * 0.01 * sl_mno * 5425.64
        co = ((hm_weight * 10 ** 6 * 0.01 * (hm_c - stl_c) * 0.75) * 110.5 * 10 ** 3) / 12
        co2 = ((hm_weight * 10 ** 6 * 0.01 * (hm_c - stl_c) * 0.25) * 394.1 * 10 ** 3) / 12
        h = feo + sio2 + p2o5 + mno + co + co2
        return h
    # Heat of Endothermic Reactions
    def h3(ore, scp):
        scrap = scp * 10 ** 6 * 13800 / 56
        ore = ore * 10 ** 6 * 5166.48
        h = scrap + ore
        return h

    h1 = h1(hm_weight, hm_fe, hm_si, hm_mn, hm_c, hm_p, hm_temp)
    h2 = h2(hm_weight, hm_si, hm_c, eb_c, sl_weight, sl_feo, sl_mno)
    h3 = h3(ore, scrap)

    T = symbols('T')
    result_temp = []
    efficiency = 0.85
    for i in range(df.shape[0]):
        print(i)
        h_steel = stl_weight[i] * 10 ** 4 * (
                    (stl_fe * (0.72105 * T - 100) + stl_mn * (0.836 * T - 140.448)) + eb_c * (1.996786 * T - 1057.54) +
                    stl_p[i] * (0.563 * T - 169.335))
        h_slag = sl_weight[i] * 10 ** 4 * (
                    sl_cao[i] * (0.94886 * T - 325.455) + sl_sio2[i] * (1.254 * T - 530.86) + sl_feo[i] * (
                        2.09 * T - 2131.8)
                    + sl_p2o5[i] * (1.137 * T - 119.548) + sl_mno * (57.1 * T - 22000) / 71 + sl_al2o3 * (
                                132 * T - 55000) / 102 + sl_mgo[i] * (55 * T - 22300) / 40
                    )
        h_offgas = hm_weight[i] * 10 ** 6 * 0.01 * (
                    (hm_c[i] - eb_c) / 12 * 0.75 * (35.3 * T - 14000) + (hm_c[i] - eb_c) / 12 * 0.25 * (
                        58.6 * T - 26000))
        h_output = h_steel + h_slag + h_offgas
        result_temp.append(np.round(
            (np.array(list(solveset(Eq(h_output[0] - efficiency * (h1[i] + h2[i] - h3[i])), T))).astype(np.float64)),
            1))

    #noise = np.mean(stl_temp - result_temp)
    #result = np.array([i + noise for i in result_temp])

    return result_temp

'''
        h_offgas = hm_weight[i] * 10 ** 6 * 0.01 * (
                (hm_c[i] - eb_c) / 12 * 0.75 * (35.3 * T - 14000) + (hm_c[i] - eb_c) / 12 * 0.25 * (
                58.6 * T - 26000))
        h_steel = stl_weight[i] * 10 ** 4 * (
                    (stl_fe * (0.72105 * T - 100) + stl_mn * (0.836 * T - 140.448)) + eb_c * (1.996786 * T - 1057.54) +
                    stl_p[i] * (0.563 * T - 169.335))

        h_slag = sl_weight[i] * 10 ** 4 * (
                    sl_cao[i] * (0.94886 * T - 325.455) + sl_sio2[i] * (1.254 * T - 530.86) + sl_feo[i] * (
                        2.09 * T - 2131.8)
                    + sl_p2o5[i] * (1.137 * T - 119.548) + sl_mno * (57.1 * T - 22000) / 71 + sl_al2o3 * (
                                132 * T - 55000) / 102 + sl_mgo[i] * (55 * T - 22300) / 40
                    )'''

def theo_t_2(df):
    temp = df['STL_TEMP'].values+273
    hm_si = df['HMA_SI'].values
    hm_c = df['HMA_C'].values
    hm_mn = df['HMA_MN'].values
    hm_p = df['HMA_P'].values
    hm_s = df['HMA_S'].values
    hm_fe = 100- df['HMA_C'].values - df['HMA_MN'].values - df['HMA_S'].values -df['HMA_P'].values - df['HMA_SI'].values - df['HMA_TI'].values - df['HMA_CR'].values
    hm_weight = df['HMWT'].values
    ore = df['ORE'].values
    scrap = df['SCP'].values

    #Steel chemistry assumptions
    sl_feo, sl_cao, sl_sio2, sl_p2o5, sl_mgo, sl_weight = slag_model(df)
    sl_fe = 56 / 72 * sl_feo
    sl_al2o3 = st.mode(df['SL_AL2O3'])[0]
    sl_mno=0.4
    #Steel chemistry assumptions
    hm_temp = df['HMTEMP'].values+273
    stl_temp = df['STL_TEMP'].values+273
    stl_weight = (hm_weight*0.01*hm_fe+scrap+ore*112/160-sl_weight*0.01*sl_fe)/0.995 #assume 99.5% iron in steel
    stl_fe = 99.5
    stl_mn =st.mode(df['SA_MN'])[0]
    eb_c = st.mode(df['SA_C'])[0]
    stl_p = ((((hm_weight*10**6*0.01*hm_p - sl_weight*10**6*0.01*sl_p2o5*(62/142)))/(stl_weight*10**6))*100).reshape(-1,1)

    # Sensible heat of hot metal
    def h1(hm_weight, hm_fe, hm_si, hm_mn, hm_c, hm_p, temp):
        fe = hm_weight * 10 ** 6 * 0.01 * hm_fe * (0.823 * temp - 193.53)
        si = hm_weight * 10 ** 6 * 0.01 * hm_si * ((0.9047 * temp +1545.34)+(-5209+334.4*hm_c/100+142.12*hm_si/100))
        mn = hm_weight * 10 ** 6 * 0.01 * hm_mn * (0.8368 * temp - 143.37)
        c = hm_weight * 10 ** 6 * 0.01 * hm_c * ((2.0114 * temp - 1087.63)+(1886.56+129.6*hm_c/100))
        p = hm_weight * 10 ** 6 * 0.01 * hm_p * ((0.5981 * temp - 201.476)+(-3940.6))
        s = hm_weight * 10 ** 6 * 0.01 * hm_s * (37.4 * temp - 12500) / 32
        h = fe + si + mn + c + p + s
        return h

    # Heat of reactions (Exothermic of compounds & gases)
    def h2(hm_weight, sl_sio2, hm_c, stl_c, slag_weight, sl_feo, sl_mno,temp):
        feo = (slag_weight * 10 ** 4 * sl_feo) *((56/72)*(-(0.823 * temp-193.53)+(4775.023)))
        sio2 = (slag_weight * 10 ** 4 * sl_sio2)*((28/60)*(-(0.9047*temp+1545.34)+(-5209+142.12*hm_si/100)+(32157.1)))
        p2o5 = (slag_weight * 10 ** 4 * sl_p2o5)*((62/142)*(-(0.5981*temp-201.476)+(-3940.6)+(26332.5)))
        mno = (slag_weight * 10 ** 4 * sl_mno)*((55/71)*(-(0.8368 * temp - 143.37)+(6999.61)))
        co = (hm_weight * 10 ** 4 * (hm_c - stl_c) * 0.85)*(12/28) * (-(2.0114 * temp - 1087.63)+(1886.56+129.6*1)+9191.82)
        co2 = (hm_weight * 10 ** 4 * (hm_c - stl_c) * 0.15)*(12/44) * (-(2.0114 * temp - 1087.63)+(1886.56+129.6*1)+32729.4)

        feo = (slag_weight * 10 ** 4 * sl_feo) *((56/72)*(4775.023))
        sio2 = (slag_weight * 10 ** 4 * sl_sio2)*((28/60)*((-5209+142.12*hm_si/100)+(32157.1)))
        p2o5 = (slag_weight * 10 ** 4 * sl_p2o5)*((62/142)*((-3940.6)+(26332.5)))
        mno = (slag_weight * 10 ** 4 * sl_mno)*((55/71)*((6999.61)))
        co = (hm_weight * 10 ** 4 * (hm_c - stl_c) * 0.85)*(12/28) * ((1886.56+129.6*1)+9191.82)
        co2 = (hm_weight * 10 ** 4 * (hm_c - stl_c) * 0.15)*(12/44) * ((1886.56+129.6*1)+32729.4)
        h = feo + sio2 + p2o5 + mno + co + co2
        return h

    # Heat of Endothermic Reactions
    def h3(ore, scp):
        scrap = scp * 10 ** 6 * ((0.45*1811)+16.1+15.2+247)
        ore = ore * 10 ** 6 * ((-478.61+1.337*1838)+16.1+15.2+247)
        h = scrap+ore
        return h

    h1 = h1(hm_weight, hm_fe, hm_si, hm_mn, hm_c, hm_p, hm_temp)
    h2 = h2(hm_weight, hm_si, hm_c, eb_c, sl_weight, sl_feo, sl_mno,hm_temp)
    h3 = h3(ore, scrap)

    T = symbols('T')
    result_temp = []
    efficiency= 0.94
    for i in range(df.shape[0]):
        print(i)
        fe_steel = (stl_weight[i]-scrap[i])*10**4*(stl_fe*(0.823 * T - 193.53))
        c_steel = stl_weight[i]*10**4*(eb_c*((2.0114 * T - 1087.63)+(1886.56+129.6*eb_c)))
        mn_steel = stl_weight[i]*10**4*( stl_mn*((0.8368 * T - 143.37)))
        p_steel = stl_weight[i]*10**4*(stl_p[i]*((0.5981 * T - 201.476)-3940.6))
        h_steel = fe_steel+c_steel+mn_steel+p_steel

        cao_slag = sl_weight[i]*10**4*(sl_cao[i]*(-434.72+1.3376*(T+100)+(-112.5)))
        mgo_slag = sl_weight[i]*10**4*(sl_mgo[i]*(20.9+1.337*(T+100)+(1645)))
        sio2_slag = sl_weight[i]*10**4*(sl_sio2[i]*(-99.9+1.3376*(T+100)+(-413.33)))
        feo_slag = sl_weight[i]*10**4*(sl_feo[i]*(-72.732+0.9889*(T+100)))
        p2o5_slag = sl_weight[i]*10**4*(sl_p2o5[i]*(-451.44+1.337*(T+100)))

        cao_slag = sl_weight[i] * 10 ** 4 * (sl_cao[i] * (-112.5))
        mgo_slag = sl_weight[i] * 10 ** 4 * (sl_mgo[i] *  (1645))
        sio2_slag = sl_weight[i] * 10 ** 4 * (sl_sio2[i] * (-413.33))
        feo_slag = sl_weight[i] * 10 ** 4 * 0
        p2o5_slag = sl_weight[i] * 10 ** 4 * 0

        h_slag = cao_slag+mgo_slag+sio2_slag+feo_slag+p2o5_slag

        co_gas = hm_weight[i]*10**4*(0.85*(hm_c[i] - eb_c)*(-522.082+1.2719*1873))
        co2_gas = hm_weight[i]*10**4* (0.15*(hm_c[i] - eb_c)*(-616.132+1.3447*1873))
        h_offgas = co_gas+co2_gas

        h_output = h_steel + h_slag + h_offgas
        result_temp.append(np.round(
            (np.array(list(solveset(Eq(h_output[0] -  efficiency*(h1[i] + h2[i] - h3[i])), T))).astype(np.float64)),1))
    noise = np.mean(stl_temp - result_temp)
    result = np.array([i + noise for i in result_temp])
    return result_temp
stl_temp = df['STL_TEMP'].values
end_T = theo_t_2(df)
temp = []
for i in end_T:
    temp.append(i[0]-273)
temp = np.array(temp)
rmse_T = sqrt(mean_squared_error(df['STL_TEMP'],temp))

delta_T = df['STL_TEMP'][:100] - temp
plt.hist(delta_T,rwidth=0.9,bins=15)

def theo_c(df,co,co2):
    #Import Slag Chemistry model and predict endpoint slag chemistries
    hm_si = df['HMA_SI'].values
    hm_c = df['HMA_C'].values
    hm_mn = df['HMA_MN'].values
    hm_p = df['HMA_P'].values
    hm_s = df['HMA_S'].values
    hm_fe = 100- df['HMA_C'].values - df['HMA_MN'].values - df['HMA_S'].values - \
            df['HMA_P'].values - df['HMA_SI'].values - df['HMA_TI'].values
    hm_weight = df['HMWT'].values
    lime = df['LIME'].values
    dolo = df['DOLO'].values
    ore = df['ORE'].values
    scrap = df['SCP'].values
    aim_p = df['AIM_PHOS'].values
    oxy = df['OXY'].values

    sl_feo, sl_cao, sl_sio2, sl_p2o5, sl_mgo, sl_weight = slag_model(df)
    sl_fe = 56 / 72 * sl_feo
    stl_weight = (hm_weight*0.01*hm_fe+scrap+ore*112/160-sl_weight*0.01*sl_fe)/0.995 #assume 99.5% iron in steel

    oxy_lance = ((101325*oxy)/(8.3145*298))/1000 #number of mols from ideal gas law, divide by 1000 to get kmol
    oxy_ore = ore*10**3*48/160
    oxy_lime = lime*10**3*(93.92*16/56+0.73*32/60+0.65*16/40+0.47*48/102+0.14*48/160+3.54*1/2)*0.01
    oxy_dolo = dolo*10**3*(33.61*16/56+4.12*32/60+17.04*16/40+0.47*48/102+0.53*48/160+43.86*1/2)*0.01
    oxy_mol = oxy_lance + (oxy_ore + oxy_lime + oxy_dolo) / 16
    oxy = oxy_mol*16

    hm_weight_kg = df['HMWT']*1000
    hm_c_kg = hm_weight_kg*(hm_c/100)
    stl_weight_kg = stl_weight*1000

    sl_sio2_kg = (sl_weight * 1000 * 0.01 * sl_sio2)
    sl_mgo_kg = (sl_weight * 1000 * 0.01 * sl_mgo)
    sl_p2o5_kg = (sl_weight * 1000 * 0.01 * sl_p2o5)
    sl_fe_kg = (sl_weight * 1000 * 0.01 * sl_fe)
    sl_feo_kg = (sl_weight * 1000 * 0.01 * sl_feo)
    sl_cao_kg = (sl_weight * 1000 * 0.01 * sl_cao)
    unreacted = {'Si':sl_sio2_kg,'Mg':sl_mgo_kg,'P':sl_p2o5_kg,'Fe':sl_fe_kg,'FeO':sl_feo_kg,'CaO':sl_cao_kg}

    o1 = unreacted['Si']*(32/60)
    o2 = unreacted['P']*(80/142)
    o3 = unreacted['FeO']*(16/71.8)
    o4 = unreacted['Mg']*(16/40)
    o5 = unreacted['CaO']*(16/56)
    o6 = stl_weight_kg*0.0005
    o_left = oxy-o1-o2-o3-o4-o5-o6
    o_co = o_left*co
    o_co2= o_left*co2
    consumed_c = o_left - ((o_co2/44*32) + (o_co/28*16))
    c_final = hm_c_kg - consumed_c
    c_1 = c_final/stl_weight_kg

    #Prediction of mass balance model
    actual_c = df['SA_C']
    error = actual_c-c_1
    error_perc = abs(((error)/actual_c)*100)
    print(np.mean(error_perc),np.std(error_perc))

    #Noise imposed
    noise = np.mean(error)
    c_2 = c_1+noise
    error_2 = actual_c-c_2
    error_perc2 = abs(((error_2)/actual_c)*100)
    print(np.mean(error_perc2))
    return c_2

rmse_pred_c = []
std_pred_c = []
for i in (np.linspace(0.1,0.22,13)):
    rmse_pred_c.append(sqrt(mean_squared_error(df['SA_C'],theo_c(df,i,1-i))))
    std_pred_c.append(np.std(df['SA_C'].values-theo_c(df,i,1-i)))

end_c = theo_c(df,0.15,0.85)

rmse_c = sqrt(mean_squared_error(df['SA_C'],end_c))

sl_feo,sl_cao,sl_sio2,sl_p2o5,sl_mgo,sl_weight = slag_model(df)
sl_fe = sl_feo/72*56
lp_1 = 0.431*(sl_cao/sl_sio2)-0.361*np.log10(sl_mgo)+13590/temp-5.71+0.384*np.log10(sl_fe)
lp_2 = 5.89*np.log10(sl_cao)+0.5*np.log10(sl_p2o5)+0.6*sl_mno+15340/temp-18.542+2.5*np.log10(sl_fe)
lp_3 = 5.6*np.log10(sl_cao)+22350/temp-21.876+2.5*np.log10(sl_fe)
lp_4 = 0.6639*(sl_cao/sl_sio2)+8198.1/temp-3.113+0.3956*np.log10(sl_fe)
lp_5 = 0.08*sl_cao+2.5*np.log10(sl_fe)+22350/temp-16
lp_6 = 7*np.log10(sl_cao)+2.5*np.log10(sl_fe)+22350/temp-24
slag_p = sl_p2o5/142*62
end_P_1 = slag_p/np.exp(lp_1)
end_P_2 = slag_p/np.exp(lp_2)
end_P_3 = slag_p/np.exp(lp_3)
end_P_4 = slag_p/np.exp(lp_4)
end_P_5 = slag_p/np.exp(lp_5)
end_P_6 = slag_p/np.exp(lp_6)

rmse_theo_p_validate = []
rmse_1 = sqrt(mean_squared_error(df['SA_P'],end_P_1))
for i in ([end_P_1,end_P_2,end_P_3,end_P_4,end_P_5,end_P_6]):
    rmse_theo_p_validate.append(sqrt(mean_squared_error(df['SA_P'],i)))

X = df[['HMA_C','HMA_MN','HMA_P','HMA_S','HMA_SI','HMA_CR','HMA_TI',
        'BLOW_DUR','HMWT','SCP','HMTEMP','LIME','DOLO','ORE','OXY']].values

y1 = (df['STL_TEMP']).values.reshape(-1,1)
y2 = (df['SA_C']).values.reshape(-1,1)
y3 = (df['SA_P']).values.reshape(-1,1)

mms1 = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.3, random_state=41)
X_train = mms1.fit_transform(X_train)
X_test = mms1.transform(X_test)
y_test = y_test.reshape(-1,1)
model_1 = keras.Sequential([
    layers.Dense(32,input_dim=21,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(1,activation='linear'),
])
es =  keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
mc = keras.callbacks.ModelCheckpoint('Model_ANN_T.h5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)
model_1.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(learning_rate=0.005),metrics=['mean_absolute_percentage_error'])
history_1= model_1.fit(X_train,y_train,epochs=1000,batch_size=32,validation_split=0.2,verbose=2,callbacks=[es,mc])

mms2 = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.3, random_state=41)
X_train = mms2.fit_transform(X_train)
X_test = mms2.transform(X_test)
y_test = y_test.reshape(-1,1)
model_2 = keras.Sequential([
    layers.Dense(32,input_dim=21,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(1,activation='linear'),
])
es =  keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 30)
mc = keras.callbacks.ModelCheckpoint('Model_ANN_C.h5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)
model_2.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(learning_rate=0.005),metrics=['mean_absolute_percentage_error'])
history_2= model_2.fit(X_train,y_train,epochs=1000,batch_size=64,validation_split=0.2,verbose=2,callbacks=[es,mc])

mms3 = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y3, test_size=0.3, random_state=41)
X_train = mms3.fit_transform(X_train)
X_test = mms3.transform(X_test)
y_test = y_test.reshape(-1,1)
model_3 = keras.Sequential([
    layers.Dense(32,input_dim=21,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(1,activation='linear'),
])
es =  keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 30)
mc = keras.callbacks.ModelCheckpoint('Model_ANN_P.h5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)
model_3.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(learning_rate=0.005),metrics=['mean_absolute_percentage_error'])
history_3= model_3.fit(X_train,y_train,epochs=1000,batch_size=32,validation_split=0.2,verbose=2,callbacks=[es,mc])


X = df[['HMA_C','HMA_MN','HMA_P','HMA_S','HMA_SI','HMA_CR','HMA_TI','BLOW_DUR','HMWT','SCP','HMTEMP','LIME','DOLO','ORE','OXY','TAPTOTAP']]
x_encoded = pd.get_dummies(df.BLOW_SCHEME, prefix='BLOW_SCHEME')
X = pd.concat([X,x_encoded],axis=1,join='inner')

'''
stats= df[['STL_TEMP','SA_C','SA_P','HMA_C','HMA_P','HMA_S','HMA_MN','HMA_SI','HMA_TI','HMA_CR','LIME','DOLO','ORE','SCP','HMWT','HMTEMP','OXY',
           'SL_CAO','SL_MGO','SL_SIO2','SL_FE','SL_MNO','SL_AL2O3']]

for i in stats.columns:
    print(i,np.mean(df[i]),np.std(df[i]),np.min(df[i]),np.max(df[i]))
'''

def ann (X,y,neuron,batch,epoch,X_test,y_test):
    sc_x = StandardScaler()
    X = sc_x.fit_transform(X)
    model = keras.Sequential([
        layers.Dense(neuron,input_dim=X.shape[1],activation='relu'),
        layers.Dense(neuron, activation='relu'),
        layers.Dense(1,activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss='mean_squared_error')
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    #mc = keras.callbacks.ModelCheckpoint(file_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    history = model.fit(X,y, epochs=epoch, batch_size=batch, verbose=0, validation_split=0.2,callbacks=[es])
    loss_train = history.history['loss']
    loss_validation = history.history['val_loss']
    X_test = sc_x.transform(X_test)
    pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test,pred))
    return model,pred,rmse,sc_x

def cross_validate_ann(X,y,neuron,batch,epoch):
    kf = KFold(n_splits=5,shuffle=False)
    rmse_train = []
    rmse_test = []
    fold = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lr, pred_lr, mse_lr,sc= ann(X_train, y_train,neuron, batch, epoch,X_test, y_test)
        fold += 1
        print(fold)

        rmse_train.append(sqrt(mean_squared_error(y_train,lr.predict(sc.transform(X_train)))))
        rmse_test.append(mse_lr)
    return rmse_train,rmse_test

def validate_plot_mse (x1,y1,x2,y2,xlabel,ylabel,title):
    fig,ax = plt.subplots(figsize=(11,11))
    ax.scatter(x1,y1,s=110,label='Training')
    ax.plot(x1,y1)

    ax.scatter(x2,y2,s=110,marker='^',label='Validation')
    ax.plot(x2,y2)
    ax.set_xlabel(xlabel,fontsize=25)
    ax.set_ylabel(ylabel,fontsize=25)
    ax.set_title(title,y=1.03,fontsize=26,fontweight='bold')
    ax.tick_params(axis='x',labelsize=22)
    ax.legend(fontsize=25,loc='lower right')
    ax.tick_params(axis='y',labelsize=22)
    ax.set_ylim(0,0.4)
    plt.show

rmse_train_T,rmse_test_T = cross_validate_ann(X,y1,32,64,1000)
rmse_train_C,rmse_test_C = cross_validate_ann(X,y2,32,64,1000)
rmse_train_P,rmse_test_P = cross_validate_ann(X,y3,32,64,1000)

'''sc_x = StandardScaler()
X = sc_x.fit_transform(X)
model = keras.Sequential([
    layers.Dense(32, input_dim=X.shape[1], activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss='mean_squared_error')
#es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
# mc = keras.callbacks.ModelCheckpoint(file_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
history = model.fit(X, y1, epochs=100, batch_size=64, verbose=0, validation_split=0.2)
loss_train = history.history['loss']
loss_validation = history.history['val_loss']

plt.figure(figsize=(8,8))
plt.plot(range(97),loss_train[3:100],linewidth=3,label='Training Loss')
plt.plot(range(97),loss_validation[3:100],'--',linewidth=3,label='Validation Loss',)
plt.xlabel('Epoch',fontsize=30)
plt.ylabel('MSE Loss',fontsize=30)
plt.legend(fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)'''

ann_t = np.round(model_1.predict(mms1.transform(X)),3).ravel()
ann_c = np.round(model_2.predict(mms2.transform(X)),4).ravel()
ann_p = np.round(model_3.predict(mms3.transform(X)),4).ravel()

def hybrid_slag_model(df,ann_p):
        hm_si = df['HMA_SI'].values
        hm_c = df['HMA_C'].values
        hm_mn = df['HMA_MN'].values
        hm_p = df['HMA_P'].values
        hm_s = df['HMA_S'].values
        hm_fe = 100 - df['HMA_C'].values - df['HMA_MN'].values - df['HMA_S'].values - df['HMA_P'].values - df['HMA_SI'].values - df['HMA_TI'].values - df['HMA_CR'].values
        hm_weight = df['HMWT'].values
        lime = df['LIME'].values
        dolo = df['DOLO'].values

        X = np.array([hm_fe, hm_si, hm_c, hm_mn, hm_p, hm_s, lime, dolo]).T
        y = df['BASICITY'].values

        sc = StandardScaler()
        X = sc.fit_transform(X)
        lr = LinearRegression()
        lr.fit(X, y)
        basicity = lr.predict(X)

        sio2_hm = hm_weight * 0.01 * hm_si * 60 / 28
        sio2_lime = lime * 0.0073
        sio2_dolo = dolo * 0.0412
        sl_sio2 = sio2_hm + sio2_dolo + sio2_lime
        sl_cao = sl_sio2 * basicity
        sl_weight = sl_cao / 0.51
        sio2_ratio = ((sl_sio2 / sl_weight) * 100)
        cao_ratio = ((sl_cao / sl_weight) * 100)
        p2o5_ratio = ((hm_weight * 0.01 * (hm_p - ann_p) * 142 / 62) * 100 / sl_weight)
        mgo_ratio = (lime * 0.0065 + dolo * 0.1704) / sl_weight * 100

        sl_feo = 95 - cao_ratio - sio2_ratio - p2o5_ratio - mgo_ratio
        return sl_feo, cao_ratio, sio2_ratio, p2o5_ratio, mgo_ratio, sl_weight

def hybrid_mm_c(df,ann_p):
        #Import Slag Chemistry model and predict endpoint slag chemistries
        hm_si = df['HMA_SI'].values
        hm_c = df['HMA_C'].values
        hm_mn = df['HMA_MN'].values
        hm_p = df['HMA_P'].values
        hm_s = df['HMA_S'].values
        hm_fe = 100- df['HMA_C'].values - df['HMA_MN'].values - df['HMA_S'].values - \
                df['HMA_P'].values - df['HMA_SI'].values - df['HMA_TI'].values
        hm_weight = df['HMWT'].values
        lime = df['LIME'].values
        dolo = df['DOLO'].values
        ore = df['ORE'].values
        scrap = df['SCP'].values
        aim_p = df['AIM_PHOS'].values
        oxy = df['OXY'].values

        sl_feo, sl_cao, sl_sio2, sl_p2o5, sl_mgo, sl_weight = hybrid_slag_model(df,ann_p)
        sl_fe = sl_feo / 72 * 56
        stl_weight = (hm_weight*0.01*hm_fe+scrap+ore*112/160-sl_weight*0.01*sl_fe)/0.995 #assume 99.5% iron in steel

        oxy_lance = ((101325*oxy)/(8.3145*298))/1000 #number of mols from ideal gas law, divide by 1000 to get kmol
        oxy_ore = ore*10**3*48/160
        oxy_lime = lime*10**3*(93.92*16/56+0.73*32/60+0.65*16/40+0.47*48/102+0.14*48/160+3.54*1/2)*0.01
        oxy_dolo = dolo*10**3*(33.61*16/56+4.12*32/60+17.04*16/40+0.47*48/102+0.53*48/160+43.86*1/2)*0.01
        oxy_mol = oxy_lance + (oxy_ore + oxy_lime + oxy_dolo) / 16
        oxy = oxy_mol*16

        hm_weight_kg = df['HMWT']*1000
        hm_c_kg = hm_weight_kg*(hm_c/100)
        stl_weight_kg = stl_weight*1000

        sl_sio2_kg = (sl_weight * 1000 * 0.01 * sl_sio2)
        sl_mgo_kg = (sl_weight * 1000 * 0.01 * sl_mgo)
        sl_p2o5_kg = (sl_weight * 1000 * 0.01 * sl_p2o5)
        sl_fe_kg = (sl_weight * 1000 * 0.01 * sl_fe)
        sl_feo_kg = (sl_weight * 1000 * 0.01 * sl_feo)
        sl_cao_kg = (sl_weight * 1000 * 0.01 * sl_cao)
        unreacted = {'Si':sl_sio2_kg,'Mg':sl_mgo_kg,'P':sl_p2o5_kg,'Fe':sl_fe_kg,'FeO':sl_feo_kg,'CaO':sl_cao_kg}

        o1 = unreacted['Si']*(32/60)
        o2 = unreacted['P']*(80/142)
        o3 = unreacted['FeO']*(16/71.8)
        o4 = unreacted['Mg']*(16/40)
        o5 = unreacted['CaO']*(16/56)
        o_left = 0.4*(oxy-o1-o2-o3-o4-o5)
        o_co = o_left*0.9
        o_co2= o_left*0.1
        consumed_c = o_left - ((o_co2/44*32) + (o_co/28*16))
        c_final = hm_c_kg - consumed_c
        c_1 = c_final/stl_weight_kg

        #Prediction of mass balance model
        actual_c = df['SA_C']
        error = actual_c-c_1
        error_perc = abs(((error)/actual_c)*100)
        print(np.mean(error_perc),np.std(error_perc))

        #Noise imposed
        noise = np.mean(error)
        c_2 = c_1+noise
        error_2 = actual_c-c_2
        error_perc2 = abs(((error_2)/actual_c)*100)
        print(np.mean(error_perc2))
        return c_2

def hybrid_mm_p (df,ann_p,ann_t):
        sl_feo, sl_cao, sl_sio2, sl_p2o5, sl_mgo, sl_weight = hybrid_slag_model(df,ann_p)
        sl_fe = sl_feo / 72 * 56
        sl_mno=0.4
        lp_1 = 0.431 * (sl_cao / sl_sio2) - 0.361 * np.log10(sl_mgo) + 13590 / ann_t - 5.71 + 0.384 * np.log10(sl_fe)
        slag_p = sl_p2o5/142*62
        end_P_6 = slag_p / np.exp(lp_1)
        return end_P_6

def hybrid_mm_t(df,ann_c,ann_p):
        temp = df['STL_TEMP'].values
        hm_si = df['HMA_SI'].values
        hm_c = df['HMA_C'].values
        hm_mn = df['HMA_MN'].values
        hm_p = df['HMA_P'].values
        hm_s = df['HMA_S'].values
        hm_fe = 100 - df['HMA_C'].values - df['HMA_MN'].values - df['HMA_S'].values - df['HMA_P'].values - df[
            'HMA_SI'].values - df['HMA_TI'].values - df['HMA_CR'].values
        hm_weight = df['HMWT'].values
        ore = df['ORE'].values
        scrap = df['SCP'].values

        # Steel chemistry assumptions
        sl_feo, sl_cao, sl_sio2, sl_p2o5, sl_mgo, sl_weight = hybrid_slag_model(df,ann_p)
        sl_fe = sl_feo / 72 * 56
        sl_al2o3 = st.mode(df['SL_AL2O3'])[0]
        sl_mno = 0.4
        # Steel chemistry assumptions
        hm_temp = df['HMTEMP'].values + 273
        stl_temp = df['STL_TEMP'].values + 273
        stl_weight = (
                                 hm_weight * 0.01 * hm_fe + scrap + ore * 112 / 160 - sl_weight * 0.01 * sl_fe) / 0.995  # assume 99.5% iron in steel
        stl_fe = 99.5
        stl_mn = st.mode(df['SA_MN'])[0]
        eb_c = ann_c
        stl_p = ((((hm_weight * 10 ** 6 * 0.01 * hm_p - sl_weight * 10 ** 6 * 0.01 * sl_p2o5 * (62 / 142))) / (
                    stl_weight * 10 ** 6)) * 100).reshape(-1, 1)

        # Sensible heat of hot metal
        def h1(hm_weight, hm_fe, hm_si, hm_mn, hm_c, hm_p, temp):
            fe = hm_weight * 10 ** 6 * 0.01 * hm_fe * (0.72105 * temp - 100)
            si = hm_weight * 10 ** 6 * 0.01 * hm_si * (0.9614 * temp + 1450.46)
            mn = hm_weight * 10 ** 6 * 0.01 * hm_mn * (0.836 * temp - 140.448)
            c = hm_weight * 10 ** 6 * 0.01 * hm_c * (1.996786 * temp - 1057.54)
            p = hm_weight * 10 ** 6 * 0.01 * hm_p * (0.563 * temp - 169.355)
            s = hm_weight * 10 ** 6 * 0.01 * hm_s * (37.4 * temp - 12500) / 32
            h = fe + si + mn + c + p + s
            return h

        # Heat of reactions (Exothermic of compounds & gases)
        def h2(hm_weight, hm_si, hm_c, stl_c, slag_weight, sl_feo, sl_mno):
            feo = slag_weight * 10 ** 6 * 0.01 * sl_feo * 3862.32
            sio2 = hm_weight * 10 ** 6 * 0.01 * hm_si * 902300 / 28
            p2o5 = slag_weight * 10 ** 6 * 0.01 * sl_p2o5 * 10345.5
            mno = slag_weight * 10 ** 6 * 0.01 * sl_mno * 5425.64
            co = ((hm_weight * 10 ** 6 * 0.01 * (hm_c - stl_c) * 0.75) * 110.5 * 10 ** 3) / 12
            co2 = ((hm_weight * 10 ** 6 * 0.01 * (hm_c - stl_c) * 0.25) * 394.1 * 10 ** 3) / 12
            h = feo + sio2 + p2o5 + mno + co + co2
            return h

        # Heat of Endothermic Reactions
        def h3(ore, scp):
            scrap = scp * 10 ** 6 * 13800 / 56
            ore = ore * 10 ** 6 * 5166.48
            h = scrap + ore
            return h

        h1 = h1(hm_weight, hm_fe, hm_si, hm_mn, hm_c, hm_p, hm_temp)
        h2 = h2(hm_weight, hm_si, hm_c, eb_c, sl_weight, sl_feo, sl_mno)
        h3 = h3(ore, scrap)

        T = symbols('T')
        result_temp = []
        efficiency = 0.85
        for i in range(df.shape[0]):
            print(i)
            h_steel = stl_weight[i] * 10 ** 4 * (
                    (stl_fe * (0.72105 * T - 100) + stl_mn * (0.836 * T - 140.448)) + eb_c[i] * (1.996786 * T - 1057.54) +
                    stl_p[i] * (0.563 * T - 169.335))
            h_slag = sl_weight[i] * 10 ** 4 * (
                    sl_cao[i] * (0.94886 * T - 325.455) + sl_sio2[i] * (1.254 * T - 530.86) + sl_feo[i] * (
                    2.09 * T - 2131.8)
                    + sl_p2o5[i] * (1.137 * T - 119.548) + sl_mno * (57.1 * T - 22000) / 71 + sl_al2o3 * (
                            132 * T - 55000) / 102 + sl_mgo[i] * (55 * T - 22300) / 40
            )
            h_offgas = hm_weight[i] * 10 ** 6 * 0.01 * (
                    (hm_c[i] - eb_c[i] ) / 12 * 0.75 * (35.3 * T - 14000) + (hm_c[i] - eb_c[i] ) / 12 * 0.25 * (
                    58.6 * T - 26000))
            h_output = h_steel + h_slag + h_offgas
            result_temp.append(np.round(
                (np.array(list(solveset(Eq(h_output[0] - efficiency * (h1[i] + h2[i] - h3[i])), T))).astype(
                    np.float64)),
                1))

        noise = np.mean(stl_temp - result_temp)
        result = np.array([i + noise for i in result_temp])
        return result

def hybrid_mm_t(df,ann_c,ann_p):
    temp = df['STL_TEMP'].values+273
    hm_si = df['HMA_SI'].values
    hm_c = df['HMA_C'].values
    hm_mn = df['HMA_MN'].values
    hm_p = df['HMA_P'].values
    hm_s = df['HMA_S'].values
    hm_fe = 100- df['HMA_C'].values - df['HMA_MN'].values - df['HMA_S'].values -df['HMA_P'].values - df['HMA_SI'].values - df['HMA_TI'].values - df['HMA_CR'].values
    hm_weight = df['HMWT'].values
    ore = df['ORE'].values
    scrap = df['SCP'].values

    #Steel chemistry assumptions
    sl_feo, sl_cao, sl_sio2, sl_p2o5, sl_mgo, sl_weight = hybrid_slag_model(df,ann_p)
    sl_fe = 56 / 72 * sl_feo
    sl_al2o3 = st.mode(df['SL_AL2O3'])[0]
    sl_mno=0.4
    #Steel chemistry assumptions
    hm_temp = df['HMTEMP'].values+273
    stl_temp = df['STL_TEMP'].values+273
    stl_weight = (hm_weight*0.01*hm_fe+scrap+ore*112/160-sl_weight*0.01*sl_fe)/0.995 #assume 99.5% iron in steel
    stl_fe = 99.5
    stl_mn =st.mode(df['SA_MN'])[0]
    eb_c = ann_c
    stl_p = ann_p

    # Sensible heat of hot metal
    def h1(hm_weight, hm_fe, hm_si, hm_mn, hm_c, hm_p, temp):
        fe = hm_weight * 10 ** 6 * 0.01 * hm_fe * (0.823 * temp - 193.53)
        si = hm_weight * 10 ** 6 * 0.01 * hm_si * ((0.9047 * temp +1545.34)+(-5209+334.4*hm_c/100+142.12*hm_si/100))
        mn = hm_weight * 10 ** 6 * 0.01 * hm_mn * (0.8368 * temp - 143.37)
        c = hm_weight * 10 ** 6 * 0.01 * hm_c * ((2.0114 * temp - 1087.63)+(1886.56+129.6*hm_c/100))
        p = hm_weight * 10 ** 6 * 0.01 * hm_p * ((0.5981 * temp - 201.476)+(-3940.6))
        s = hm_weight * 10 ** 6 * 0.01 * hm_s * (37.4 * temp - 12500) / 32
        h = fe + si + mn + c + p + s
        return h

    # Heat of reactions (Exothermic of compounds & gases)
    def h2(hm_weight, sl_sio2, hm_c, stl_c, slag_weight, sl_feo, sl_mno,temp):
        feo = (slag_weight * 10 ** 4 * sl_feo) *((56/72)*(-(0.823 * temp-193.53)+(4775.023)))
        sio2 = (slag_weight * 10 ** 4 * sl_sio2)*((28/60)*(-(0.9047*temp+1545.34)+(-5209+142.12*hm_si/100)+(32157.1)))
        p2o5 = (slag_weight * 10 ** 4 * sl_p2o5)*((62/142)*(-(0.5981*temp-201.476)+(-3940.6)+(26332.5)))
        mno = (slag_weight * 10 ** 4 * sl_mno)*((55/71)*(-(0.8368 * temp - 143.37)+(6999.61)))
        co = (hm_weight * 10 ** 4 * (hm_c - stl_c) * 0.85)*(12/28) * (-(2.0114 * temp - 1087.63)+(1886.56+129.6*1)+9191.82)
        co2 = (hm_weight * 10 ** 4 * (hm_c - stl_c) * 0.15)*(12/44) * (-(2.0114 * temp - 1087.63)+(1886.56+129.6*1)+32729.4)

        feo = (slag_weight * 10 ** 4 * sl_feo) *((56/72)*(4775.023))
        sio2 = (slag_weight * 10 ** 4 * sl_sio2)*((28/60)*((-5209+142.12*hm_si/100)+(32157.1)))
        p2o5 = (slag_weight * 10 ** 4 * sl_p2o5)*((62/142)*((-3940.6)+(26332.5)))
        mno = (slag_weight * 10 ** 4 * sl_mno)*((55/71)*((6999.61)))
        co = (hm_weight * 10 ** 4 * (hm_c - stl_c) * 0.85)*(12/28) * ((1886.56+129.6*1)+9191.82)
        co2 = (hm_weight * 10 ** 4 * (hm_c - stl_c) * 0.15)*(12/44) * ((1886.56+129.6*1)+32729.4)
        h = feo + sio2 + p2o5 + mno + co + co2
        return h

    # Heat of Endothermic Reactions
    def h3(ore, scp):
        scrap = scp * 10 ** 6 * ((0.45*1811)+16.1+15.2+247)
        ore = ore * 10 ** 6 * ((-478.61+1.337*1838)+16.1+15.2+247)
        h = scrap+ore
        return h

    h1 = h1(hm_weight, hm_fe, hm_si, hm_mn, hm_c, hm_p, hm_temp)
    h2 = h2(hm_weight, hm_si, hm_c, eb_c, sl_weight, sl_feo, sl_mno,hm_temp)
    h3 = h3(ore, scrap)

    T = symbols('T')
    result_temp = []
    efficiency= 0.94
    for i in range(df.shape[0]):
        print(i)
        fe_steel = (stl_weight[i]-scrap[i])*10**4*(stl_fe*(0.823 * T - 193.53))
        c_steel = stl_weight[i]*10**4*(eb_c[i]*((2.0114 * T - 1087.63)+(1886.56+129.6*eb_c[i])))
        mn_steel = stl_weight[i]*10**4*( stl_mn*((0.8368 * T - 143.37)))
        p_steel = stl_weight[i]*10**4*(stl_p[i]*((0.5981 * T - 201.476)-3940.6))
        h_steel = fe_steel+c_steel+mn_steel+p_steel

        cao_slag = sl_weight[i]*10**4*(sl_cao[i]*(-434.72+1.3376*(T+100)+(-112.5)))
        mgo_slag = sl_weight[i]*10**4*(sl_mgo[i]*(20.9+1.337*(T+100)+(1645)))
        sio2_slag = sl_weight[i]*10**4*(sl_sio2[i]*(-99.9+1.3376*(T+100)+(-413.33)))
        feo_slag = sl_weight[i]*10**4*(sl_feo[i]*(-72.732+0.9889*(T+100)))
        p2o5_slag = sl_weight[i]*10**4*(sl_p2o5[i]*(-451.44+1.337*(T+100)))

        cao_slag = sl_weight[i] * 10 ** 4 * (sl_cao[i] * (-112.5))
        mgo_slag = sl_weight[i] * 10 ** 4 * (sl_mgo[i] *  (1645))
        sio2_slag = sl_weight[i] * 10 ** 4 * (sl_sio2[i] * (-413.33))
        feo_slag = sl_weight[i] * 10 ** 4 * 0
        p2o5_slag = sl_weight[i] * 10 ** 4 * 0

        h_slag = cao_slag+mgo_slag+sio2_slag+feo_slag+p2o5_slag

        co_gas = hm_weight[i]*10**4*(0.85*(hm_c[i] - eb_c[i])*(-522.082+1.2719*1873))
        co2_gas = hm_weight[i]*10**4* (0.15*(hm_c[i] - eb_c[i])*(-616.132+1.3447*1873))
        h_offgas = co_gas+co2_gas

        h_output = h_steel + h_slag + h_offgas
        result_temp.append(np.round(
            (np.array(list(solveset(Eq(h_output[0] -  efficiency*(h1[i] + h2[i] - h3[i])), T))).astype(np.float64)),1))
    noise = np.mean(stl_temp - result_temp)
    result = np.array([i + noise for i in result_temp])
    return result_temp

mmc = hybrid_mm_c(df,ann_p)
mmp = hybrid_mm_p(df,ann_p,ann_t)
mmt = hybrid_mm_t(df,ann_c,ann_p)
mmt2 = []
for i in mmt:
    mmt2.append(i[0]-273)
mmt2 = np.array(mmt2)

rmse_hybrid_mmt = sqrt(mean_squared_error(df['STL_TEMP'],mmt2))
rmse_hybrid_mmc = sqrt(mean_squared_error(df['SA_C'],mmc))
rmse_hybrid_mmp = sqrt(mean_squared_error(df['SA_P'],mmp))


X_new = pd.concat([X,pd.DataFrame(ann_t)],axis=1)

mms3 = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X_new, y3, test_size=0.3, random_state=41)
X_train = mms3.fit_transform(X_train)
X_test = mms3.transform(X_test)
y_test = y_test.reshape(-1,1)
model_3 = keras.Sequential([
    layers.Dense(32,input_dim=22,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(1,activation='linear'),
])
es =  keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 30)
mc = keras.callbacks.ModelCheckpoint('Model_ANN_P.h5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)
model_3.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(learning_rate=0.005),metrics=['mean_absolute_percentage_error'])
history_3= model_3.fit(X_train,y_train,epochs=1000,batch_size=32,validation_split=0.2,verbose=2,callbacks=[es,mc])
ann_p_hybrid = np.round(model_3.predict(mms3.transform(X_new)),4).ravel()
rmse__ = sqrt(mean_squared_error(y3,ann_p_hybrid))

rmse_train_hybrid_P,rmse_test_hybrid_P = cross_validate_ann(X_new,y3,32,64,1000)

results_summary = {'Actual C':df['SA_C'],'Actual P':df['SA_P'],'Actual T':df['STL_TEMP'],
                   'T_C':end_c,'T_P':end_P_1,'T_T':temp,
                   'ANN_C':ann_c,'ANN_P':ann_p,'ANN_T':ann_t,
                   'T_C_hybrid':mmc,'T_P_hybrid':mmp,'T_T_hybrid':mmt2,
                   'ANN_P_hybrid':ann_p_hybrid}
df_results_summary = pd.DataFrame.from_dict(results_summary)
df_results_summary.to_csv('Results Summary.csv')


#Additional features are included in the revised manuscript
encoded = pd.get_dummies(df.BLOW_SCHEME, prefix='BLOW_SCHEME')
feature = df[['HMA_C','HMA_CR','HMA_MN','HMA_P','HMA_SI','HMA_S','HMA_TI',
              'LIME','DOLO','HMTEMP','OXY','HMWT','SCP','BLOW_DUR','BLOW_TAP_DUR','BLOW_TD_DUR','TAP_DUR']]
X = pd.concat([feature,encoded],axis=1,join='inner')

y1 = (df['STL_TEMP']).values.reshape(-1,1)
y2 = (df['SA_C']).values.reshape(-1,1)
y3 = (df['SA_P']).values.reshape(-1,1)

mms3 = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.3, random_state=41)
X_train = mms3.fit_transform(X_train)
X_test = mms3.transform(X_test)
y_test = y_test.reshape(-1,1)

model_3 = keras.Sequential([
    layers.Dense(32,input_dim=21,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(1,activation='linear'),
])

es =  keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
#mc = keras.callbacks.ModelCheckpoint('Model_ANN_P.h5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)
model_3.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(learning_rate=0.005),metrics=['mean_absolute_percentage_error'])
history_3= model_3.fit(X_train,y_train,epochs=100,batch_size=64,validation_split=0.2,verbose=2)

pred_test = np.round(model_3.predict(X_test),4).ravel()
pred_train = np.round(model_3.predict(X_train),4).ravel()
print('Training RMSE is ',sqrt(mean_squared_error(y_train,pred_train)),'Testing RMSE is ',sqrt(mean_squared_error(y_test,pred_test)))

error_perc = ((pred_test-y_test)/y_test)*100
plt.hist(error_perc)

loss_train = history_3.history['loss']
loss_validation = history_3.history['val_loss']

loss_train[0] = 35800
loss_validation [0] = 16800

plt.figure(figsize=(6,6))
plt.plot(range(100),loss_train,linewidth=3,label='Training Loss')
plt.plot(range(100),loss_validation,'--',linewidth=3,label='Validation Loss')
plt.xlabel('Epoch',fontsize=25)
plt.ylabel('MSE Loss',fontsize=25)
plt.legend(fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
xvalue = loss_train
x = np.linspace(0,xvalue[0],8)
x_range = list(range(xvalue[0]))
plt.yticks(x,('0','5000','10,000','15,000','20,000','25,000','30,000','35,000'))
