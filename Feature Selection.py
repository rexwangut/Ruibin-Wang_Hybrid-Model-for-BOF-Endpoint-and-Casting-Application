import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import *
from math import sqrt
from sklearn.feature_selection import SelectKBest,mutual_info_regression
import seaborn as sns
import statsmodels.api as sm

#1 Baseline Model (No Feature Selection Technique implemented)
def rmse (pred,actual):
    rmse = sqrt(mean_squared_error(actual,pred))
    return rmse
def mape (pred,actual):
    mape = np.mean((abs(actual-pred)/actual)*100)
    return mape
def hit_rate(pred,actual):
    error = actual.reshape(-1,1) - pred
    within15 = np.count_nonzero(abs(error)<=15)*100/len(actual)
    return within15
def ann (X,y,neuron,batch,epoch,X_test,y_test):
    sc_x = StandardScaler()
    X = sc_x.fit_transform(X)
    model = keras.Sequential([
        layers.Dense(neuron,input_dim=X.shape[1],activation='relu'),
        layers.Dense(1,activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    #mc = keras.callbacks.ModelCheckpoint(file_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    history = model.fit(X,y, epochs=epoch, batch_size=batch, verbose=2, validation_split=0.2,callbacks=[es])
    X_test = sc_x.transform(X_test)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test,pred)
    r2 = r2_score(y_test,pred)
    return model,pred,r2,mse,sc_x
def cross_validate_ann(X,y,neuron,batch,epoch):
    kf = KFold(n_splits=5,shuffle=False)
    train_rmse_mape = []
    test_rmse_mape = []
    train_hr = []
    test_hr = []

    num = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lr, pred_lr, r2_lr, mse_lr,sc = ann(X_train, y_train,neuron, batch, epoch,X_test, y_test)
        num = num+1
        print('Fold Number: ',num)
        train_rmse_mape.append(rmse(y_train, lr.predict(sc.transform(X_train))))
        train_rmse_mape.append(mape(y_train, lr.predict(sc.transform(X_train))))
        test_rmse_mape.append(rmse(y_test,pred_lr))
        test_rmse_mape.append(mape(y_test,pred_lr))

        train_hr.append(hit_rate(y_train,lr.predict(sc.transform(X_train))))
        test_hr.append(hit_rate(y_test,pred_lr))
    return train_rmse_mape,test_rmse_mape,train_hr,test_hr

df = pd.read_csv('C:/University of Toronto/Modelling in Steelmaking/Data/Paper #2 Hybrid Model/Dataset.csv')
df.dropna(subset=['VSL_LIFE_NEW','AIM_TEMP1','DS_BLOW'],inplace=True)

df['POSTSTIRRING'].fillna('FALSE',inplace=True)
encoded_1 = pd.get_dummies(df.POSTSTIRRING,prefix='POSTSTIRRING')
encoded_2 = pd.get_dummies(df.BLOW_SCHEME, prefix='BLOW_SCHEME')


feature_og1 = df[['HMA_C','HMA_P','HMA_S','HMA_MN','HMA_SI','HMA_TI','HMA_CR',
              'LIME','DOLO','ORE','SCP','REMSTL','WRPSCP','PLDIRN',
                'HMWT','HMTEMP','BATHHT','VSL_LIFE_NEW','OXY',
                'AIM_PHOS','AIM_TEMP1','BLOW_DUR','BLOW_TD_DUR','BLOW_TAP_DUR','TAP_DUR','DS_BLOW']]

feature_og = pd.concat([feature_og1,encoded_1,encoded_2],axis=1,join='inner')

feature_pc = pd. concat([feature_og,df['EB_C'],df['EB_PHOS'],df['STL_TEMP']],axis=1,join='inner')
for i in range (feature_og.shape[1]):
    print(i,np.isnan(feature_og.iloc[:,i]).any())

corr = feature_pc.corr()

f, ax = plt.subplots(figsize=(16,16))
sns.heatmap(corr,annot=True,annot_kws={'size':6},fmt='.1g',vmin=-1,vmax=1,linecolor='white',linewidth=1,square=True)
plt.show()

#2 Mutual information (Filter method)
mms = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(feature_og,df['SA_C'].values, test_size=0.3, random_state=42)
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)
y_test = y_test.reshape(-1,1)


for i in range (feature_og.shape[1]):
    print(i,np.isnan(feature_og.iloc[:,i]).any())

def mutual_info (X_train,y_train,X_test):
    fs = SelectKBest(score_func = mutual_info_regression,k=16)
    fs.fit(X_train,y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs,X_test_fs,fs

X_train_fs,X_test_fs,fs = mutual_info(X_train,y_train,X_test)

col = fs.get_support(indices=True)
features_mutual_info =feature_og.iloc[:,col]

for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))

temp = pd.DataFrame(fs.scores_,range(len(fs.scores_)))
temp = temp[temp>np.mean(fs.scores_)]
temp.dropna(inplace=True)
temp.sort_values(by=[0],inplace = True,ascending=False)

x = np.arange(0, 32, 1)
from matplotlib import pyplot
from sklearn.pipeline import Pipeline
# plot the scores
fig,ax = plt.subplots(figsize=(11,11))
ax.bar([i for i in range(len(fs.scores_))], fs.scores_)
ax.plot([0, 32], [np.mean(fs.scores_)/np.max(fs.scores_),np.mean(fs.scores_)/np.max(fs.scores_)], color='red',linewidth=4,transform=ax.transAxes)
ax.set_xlabel('Feature Index',fontsize=20)
ax.set_ylabel('Mutual Information Score',fontsize=20)
ax.set_title('Mutual Information Score for Endpoint Carbon',y=1.02,fontsize=20,fontweight='bold')
ax.tick_params(axis='x',labelsize=22)
ax.tick_params(axis='y',labelsize=22)
ax.set_xticks(x)
plt.show


#Random Forest Regressor Feature Selection
mms = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(feature_og,df['STL_TEMP'].values, test_size=0.3, random_state=42)
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)
y_test = y_test.reshape(-1,1)

rf = RandomForestRegressor(n_estimators=200)
rf.fit(X_train,y_train)
#rf.feature_importances_
plt.figure(figsize=(15,10))
sorted_idx = rf.feature_importances_.argsort()
plt.barh(feature_og.columns[sorted_idx], rf.feature_importances_[sorted_idx])
ratio = np.mean(rf.feature_importances_)/np.max(rf.feature_importances_)
plt.plot([ratio,ratio],[0,32],color='red',linewidth=4,transform=ax.transAxes)
plt.xlabel("Feature Importance",fontsize=18)
plt.title('Random Forest Feature Importance for Endpoint Temperature',y=1.02,fontsize=20,fontweight='bold')

rf.feature_importances_[sorted_idx]

print(np.mean(rf.feature_importances_))

#Stepwise Feature Selection
mms = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(feature_og,df['SA_P'].values, test_size=0.3, random_state=42)
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)
y_test = y_test.reshape(-1,1)


def stepwise_selection(X, y,initial_list=[],threshold_in=0.01,threshold_out=0.05,verbose=True):
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

result = stepwise_selection(feature_og, df['STL_TEMP'])

print('resulting features:')
print(result)

#Princial Component Analysis
X = feature_og.values
y = df['STL_TEMP'].values
X = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA()
X_pca = pca.fit_transform(X)
exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

plt.figure(figsize=(12,10))
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio',fontsize=25,fontweight='bold')
plt.xlabel('Principal component index',fontsize=25,fontweight='bold')
plt.xticks(fontsize=15,fontweight='bold')
plt.yticks(fontsize=15,fontweight='bold')
plt.plot([0,32],[0.95,0.95],color='red',linewidth=2,transform=ax.transAxes,linestyle='dotted')
plt.legend(loc='best',fontsize=18)
plt.title('Principal Component Analysis Among Input Features',fontsize=25,fontweight='bold',y=1.03)
plt.show()


#Feature Subset Comparison
import time
def ann (X,y,neuron,batch,epoch,X_test,y_test):
    start = time.time()
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
    history = model.fit(X,y, epochs=epoch, batch_size=batch, verbose=2, validation_split=0.2,callbacks=[es])
    X_test = sc_x.transform(X_test)
    pred = model.predict(X_test).reshape(-1,1)
    rmse = sqrt(mean_squared_error(y_test,pred))
    end = time.time()
    duration = end-start
    error = y_test-pred
    std = np.std(error)
    return rmse,duration,std

feature_og1 = df[['HMA_C','HMA_P','HMA_S','HMA_MN','HMA_SI','HMA_TI','HMA_CR',
              'LIME','DOLO','ORE','SCP','REMSTL','WRPSCP','PLDIRN',
                'HMWT','HMTEMP','BATHHT','VSL_LIFE_NEW','OXY',
                'AIM_PHOS','AIM_TEMP1','BLOW_DUR','BLOW_TD_DUR','BLOW_TAP_DUR','TAP_DUR','DS_BLOW']]

feature_og = pd.concat([feature_og1,encoded_1,encoded_2],axis=1,join='inner')

encoded_1 = pd.get_dummies(df.POSTSTIRRING,prefix='POSTSTIRRING')
encoded_2 = pd.get_dummies(df.BLOW_SCHEME, prefix='BLOW_SCHEME')

C_pc = df[['HMA_SI','HMA_CR','SCP','WRPSCP','HMWT','VSL_LIFE_NEW','OXY','TAP_DUR']]
C_pc= pd.concat([C_pc,encoded_1['POSTSTIRRING_FALSE']],axis=1,join='inner')
P_pc = df[['HMA_SI','LIME','WRPSCP','PLDIRN','HMWT','HMTEMP','OXY','AIM_PHOS','AIM_TEMP1','BLOW_TD_DUR','BLOW_TAP_DUR','TAP_DUR',]]
T_pc = df[['HMA_P','HMA_SI','LIME','REMSTL','WRPSCP','HMWT','HMTEMP','BATHHT','VSL_LIFE_NEW','OXY','AIM_TEMP1','BLOW_DUR','BLOW_TD_DUR','BLOW_TAP_DUR']]
T_pc= pd.concat([T_pc,encoded_2['BLOW_SCHEME_2'],encoded_2['BLOW_SCHEME_4']],axis=1,join='inner')

X_train, X_test, y_train, y_test = train_test_split(C_pc, df['SA_C'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_1,duration_1,std_1 = ann(X_train,y_train,32,64,1000,X_test,y_test)

X_train, X_test, y_train, y_test = train_test_split(P_pc, df['SA_P'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_2,duration_2,std_2 = ann(X_train,y_train,32,64,1000,X_test,y_test)

X_train, X_test, y_train, y_test = train_test_split(T_pc, df['STL_TEMP'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_3,duration_3,std_3 = ann(X_train,y_train,32,64,1000,X_test,y_test)


C_mi = df[['BATHHT','HMA_S','ORE','AIM_TEMP1','HMA_SI','HMA_C','HMA_P','SCP','HMA_CR','HMA_MN','VSL_LIFE_NEW','TAP_DUR']]
P_mi = df[['LIME','DOLO','AIM_TEMP1','BLOW_TAP_DUR','OXY','HMA_SI','VSL_LIFE_NEW','ORE','HMTEMP','HMA_TI','HMA_S','SCP']]
T_mi = df[['AIM_TEMP1','BLOW_TAP_DUR','OXY','ORE','DOLO','HMA_SI','BLOW_DUR','AIM_PHOS','VSL_LIFE_NEW','LIME','HMTEMP','SCP','HMA_TI']]

X_train, X_test, y_train, y_test = train_test_split(C_mi, df['SA_C'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_1,duration_1,std_1 = ann(X_train,y_train,32,64,1000,X_test,y_test)

X_train, X_test, y_train, y_test = train_test_split(P_mi, df['SA_P'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_2,duration_2,std_2 = ann(X_train,y_train,32,64,1000,X_test,y_test)

X_train, X_test, y_train, y_test = train_test_split(T_mi, df['STL_TEMP'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_3,duration_3,std_3 = ann(X_train,y_train,32,64,1000,X_test,y_test)


C_step = df[['BATHHT','ORE','TAP_DUR','HMA_S','OXY','HMTEMP','SCP','DS_BLOW','HMA_TI','BLOW_TAP_DUR','LIME','HMA_CR','BLOW_DUR','WRPSCP']]
C_step= pd.concat([C_step,encoded_1['POSTSTIRRING_True'],encoded_2['BLOW_SCHEME_2']],axis=1,join='inner')

P_step = df[['LIME','AIM_TEMP1','HMWT','ORE','BLOW_TAP_DUR','BLOW_TD_DUR','DOLO','AIM_PHOS','HMA_S','HMA_P','OXY','PLDIRN','DS_BLOW','TAP_DUR','VSL_LIFE_NEW','BLOW_DUR',
             'HMTEMP','WRPSCP','HMA_CR','HMA_SI','BATHHT']]
P_step = pd.concat([P_step,encoded_2['BLOW_SCHEME_3']],axis=1,join='inner')

T_step = df[['AIM_TEMP1','OXY','SCP','ORE','DOLO','HMTEMP','HMA_TI','DS_BLOW','BLOW_TD_DUR','BLOW_TAP_DUR','LIME','HMA_SI','HMA_C','WRPSCP','BLOW_DUR',
             'BATHHT','HMA_S','HMA_P','AIM_PHOS','HMA_MN']]
T_step = pd.concat([T_step,encoded_1,encoded_2[['BLOW_SCHEME_2','BLOW_SCHEME_4']]],axis=1,join='inner')

X_train, X_test, y_train, y_test = train_test_split(C_step, df['SA_C'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_1,duration_1,std_1 = ann(X_train,y_train,32,64,1000,X_test,y_test)

X_train, X_test, y_train, y_test = train_test_split(P_step, df['SA_P'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_2,duration_2,std_2 = ann(X_train,y_train,32,64,1000,X_test,y_test)

X_train, X_test, y_train, y_test = train_test_split(T_step, df['STL_TEMP'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_3,duration_3,std_3 = ann(X_train,y_train,32,64,1000,X_test,y_test)


C_rf = df[['VSL_LIFE_NEW','BATHHT','ORE','OXY','DOLO','DS_BLOW','LIME','HMTEMP','HMA_P','HMA_SI','HMA_S','HMA_C','HMA_TI','BLOW_DUR','HMWT','HMA_CR','TAP_DUR']]

P_rf = df[['LIME','OXY','HMWT','ORE','BLOW_TAP_DUR','DOLO','VSL_LIFE_NEW','DS_BLOW','HMTEMP','HMA_S','HMA_P','HMA_TI','HMWT','HMA_C','HMA_MN']]

T_rf = df[['AIM_TEMP1','OXY','SCP','ORE','DOLO','BLOW_TAP_DUR','HMTEMP','LIME','VSL_LIFE_NEW','BLOW_TD_DUR','HMA_SI']]

X_train, X_test, y_train, y_test = train_test_split(C_rf, df['SA_C'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_1,duration_1,std_1 = ann(X_train,y_train,32,64,1000,X_test,y_test)

X_train, X_test, y_train, y_test = train_test_split(P_rf, df['SA_P'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_2,duration_2,std_2 = ann(X_train,y_train,32,64,1000,X_test,y_test)

X_train, X_test, y_train, y_test = train_test_split(T_rf, df['STL_TEMP'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_3,duration_3,std_3 = ann(X_train,y_train,32,64,1000,X_test,y_test)

X_pca = X_pca[:,:22]
X_train, X_test, y_train, y_test = train_test_split(X_pca , df['SA_C'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_1,duration_1,std_1 = ann(X_train,y_train,32,64,1000,X_test,y_test)

X_train, X_test, y_train, y_test = train_test_split(X_pca , df['SA_P'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_2,duration_2,std_2 = ann(X_train,y_train,32,64,1000,X_test,y_test)

X_train, X_test, y_train, y_test = train_test_split(X_pca , df['STL_TEMP'].values, test_size=0.3, random_state=42)
y_test = y_test.reshape(-1,1)
rmse_3,duration_3,std_3 = ann(X_train,y_train,32,64,1000,X_test,y_test)

a = C_mi.columns.intersection(P_rf.columns)
