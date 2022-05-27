import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import concat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
import keras
from keras import layers
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from math import sqrt

def mse (actual,pred):
    print ('mse is :',mean_squared_error(actual,pred))
    return mean_squared_error(actual,pred)

def mape (y,pred):
    mape=[]
    for i in range(y.shape[0]):
        mape.append((abs(y[i]-pred[i])/y[i])*100)
    return np.sum(mape)/(y.shape[0])

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def normalize (data,n_in):
    sc_x = MinMaxScaler(feature_range=(0, 1))
    sc_y = MinMaxScaler(feature_range=(0, 1))
    X = sc_x.fit_transform(data.iloc[:,:n_in])
    y = sc_y.fit_transform(data.iloc[:, -1].values.reshape(-1, 1))
    return X,y,sc_x,sc_y

def split (X,y,perc):
    X_train = X[:int(X.shape[0] * perc)]
    X_test = X[int(X.shape[0] * perc):]
    y_train = y[:int(X.shape[0] * perc)]
    y_test = y[int(X.shape[0] * perc):]
    return X_train,X_test,y_train,y_test

def reshape(training_data,num_feature):
    data = np.reshape(training_data,(training_data.shape[0],training_data.shape[1],num_feature))
    return data

def inverse (x,sc):
    return sc.inverse_transform(x)

def plot_compare(title,ylabel,xlabel,range1,y1,y2,color1,color2,label1,label2):
    plt.figure(figsize=(18,10))
    plt.title(title, fontsize=25,y=0.93,fontweight='bold')
    plt.ylabel(ylabel, fontsize=25)
    plt.xlabel(xlabel, fontsize=25)
    #plt.xticks(np.arange(0,len(range1),4),fontsize=22)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.plot(range1, y1, color1, label=label1,linewidth=1.3)
    plt.plot(range1, y2, color2, label=label2,linestyle='dotted',linewidth=0.8)
    plt.legend(fontsize=25,loc='lower right')

df = pd.read_csv('C:/University of Toronto/Modelling in Steelmaking/Data/Stopper Rod (Stelco)/Data/Final Datasets/ULC.csv')

values_1 = df[['CI1']].values
values_2 = df[['CI2']].values
values_3 = df[['CI3']].values

#By using Clogging Index 1 for Time Series Forecasting
n_in = 60
n_out = 12

data_1 = series_to_supervised(values_1,n_in,n_out,True)
data_2 = series_to_supervised(values_2,n_in,n_out,True)
data_3 = series_to_supervised(values_3,n_in,n_out,True)
'''
n_out=12
srp = df[['STOP_ROD_POS_ST1']].values
csp = df[['CAST_SPD_ST1']].values
srp_ts = series_to_supervised(srp,n_in,n_out,True)
srp_ts = srp_ts.iloc[:,:-n_out]
csp_ts = series_to_supervised(csp,n_in,n_out,True)
csp_ts = csp_ts.iloc[:,:-n_out]
ts = pd.concat([srp_ts,csp_ts],axis=1)
ts['CI'] = values_2[n_out+59:]
X_1,y_1,sc_x_1,sc_y_1 = normalize(ts,n_in)
X_train_1,X_test_1,y_train_1,y_test_1 = split(X_1,y_1,0.7)
'''
X_1,y_1,sc_x_1,sc_y_1 = normalize(data_1,n_in)
X_2,y_2,sc_x_2,sc_y_2 = normalize(data_2,n_in)
X_3,y_3,sc_x_3,sc_y_3 = normalize(data_3,n_in)

#LSTM input dimension: A 3D tensor with shape [batch, timesteps, feature].

X_train_1,X_test_1,y_train_1,y_test_1 = split(X_1,y_1,0.7)
X_train_2,X_test_2,y_train_2,y_test_2 = split(X_2,y_2,0.7)
X_train_3,X_test_3,y_train_3,y_test_3 = split(X_3,y_3,0.7)

tscv = TimeSeriesSplit(n_splits=3)
def cv_split (X,y):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for train_index, test_index in tscv.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train.append(X[train_index])
        X_test.append(X[test_index])
        y_train.append(y[train_index])
        y_test.append(y[test_index])
    return {'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test}
def parameter_tuning (X_train,y_train,X_test,y_test,lr,neuron,dropout,epoch,batch,sc_y):
    model = keras.Sequential([
        layers.LSTM(units=neuron, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        layers.Dropout(dropout),
        layers.LSTM(units=neuron),
        layers.Dropout(dropout),
        layers.Dense(units=1)]
    )
    '''
    es =  keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = patience)
    mc = keras.callbacks.ModelCheckpoint(file_name, monitor='val_loss', mode='min', verbose=2, save_best_only=True)
  
    
    if patience != 0:
        history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch, verbose=2)
    else:
        history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch, verbose=2)  
    '''
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch, verbose=2,validation_split = 0.2,callbacks=[es])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred = model.predict(X_test)
    actual = sc_y.inverse_transform(y_test)
    prediction = sc_y.inverse_transform(pred)
    return model,actual,prediction,mse(actual,prediction),r2_score(actual,prediction),mape(actual,prediction)
def cv_tuning (cv_result,lr,neuron,dropout,epoch,batch,sc_y):
    parameters = [lr,neuron,dropout,epoch,batch]
    mse_tune = 0
    r2_tune = 0
    mape_tune = 0
    for i in range (3):
        print(i+1,parameters)
        X_train = cv_result['X_train'][i]
        X_test = cv_result['X_test'][i]
        y_train = cv_result['y_train'][i]
        y_test = cv_result['y_test'][i]
        X_train = reshape(X_train, 1)
        tune = parameter_tuning(X_train,y_train,X_test,y_test,lr,neuron,dropout,epoch,batch,sc_y)
        mse_tune += tune[-3]
        r2_tune += tune[-2]
        mape_tune += tune[-1]
    mse_average = mse_tune/3
    r2_average = r2_tune/3
    mape_average = mape_tune/3
    return {'Parameters':parameters,'Average MSE':mse_average,'Average R2':r2_average,'Average MAPE':mape_average}

#We will focus on clogging index #1 only for now
cv = cv_split(X_train_2,y_train_2)
result_1 = cv_tuning(cv,0.01,128,0.1,50,256,sc_y_1)
result_2 = cv_tuning(cv,0.01,128,0.2,50,256,sc_y_1)
result_3 = cv_tuning(cv,0.01,128,0.3,50,256,sc_y_1)
result_4 = cv_tuning(cv,0.02,128,0.1,50,256,sc_y_1)
result_5 = cv_tuning(cv,0.02,128,0.2,50,256,sc_y_1)
result_6 = cv_tuning(cv,0.02,128,0.3,50,256,sc_y_1)
result_7 = cv_tuning(cv,0.005,128,0.1,50,256,sc_y_1)
result_8 = cv_tuning(cv,0.005,128,0.2,50,256,sc_y_1)
result_9 = cv_tuning(cv,0.005,128,0.3,50,256,sc_y_1)
result_10 = cv_tuning(cv,0.01,128,0.1,50,512,sc_y_1)
result_11 = cv_tuning(cv,0.01,128,0.2,50,512,sc_y_1)
result_12 = cv_tuning(cv,0.01,128,0.3,50,512,sc_y_1)
result_13 = cv_tuning(cv,0.02,128,0.1,50,512,sc_y_1)
result_14 = cv_tuning(cv,0.02,128,0.2,50,512,sc_y_1)
result_15 = cv_tuning(cv,0.02,128,0.3,50,512,sc_y_1)
result_16 = cv_tuning(cv,0.005,128,0.1,50,512,sc_y_1)
result_17 = cv_tuning(cv,0.005,128,0.2,50,512,sc_y_1)
result_18 = cv_tuning(cv,0.005,128,0.3,50,512,sc_y_1)

def lstm_training (X_train,y_train,X_test,y_test,lr,neuron,dropout,epoch,batch,sc_y):
    model = keras.Sequential([
        layers.LSTM(units=neuron, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        layers.Dropout(dropout),
        layers.LSTM(units=neuron),
        layers.Dropout(dropout),
        layers.Dense(units=1)]
    )
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch, verbose=2,validation_data = (reshape(X_test,1),y_test),shuffle=1,callbacks=[es])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred = model.predict(X_test)
    actual = sc_y.inverse_transform(y_test)
    prediction = sc_y.inverse_transform(pred)
    return model,actual,prediction,mse(actual,prediction),r2_score(actual,prediction),mape(actual,prediction)

lr = 0.01
neuron = 128
dropout = 0.3
epoch = 50
batch = 256

X_train_1 = reshape(X_train_2,1)

model_final,actual,prediction,mse_lstm,r2_lstm,mape_lstm = lstm_training(X_train_1,y_train_2,X_test_2,y_test_2,lr,neuron,dropout,epoch,batch,sc_y_1)

plot_compare('ULC Clogging Index 2','Clogging Index','Datapoints',range(len(actual)),actual,prediction,'g','r','Actual Value','Predicted Value')
plot_compare('Clogging Index 1','Clogging Index','Datapoints',range(len(actual)),sc_y_1.transform(actual),sc_y_1.transform(prediction),'g','r','Actual Value','Predicted Value')

plot_compare('Clogging Index 1','Clogging Index','Datapoints',range(len(actual[63350:63400])),actual[63350:63400],prediction[63350:63400],'g','r','Actual Value','Predicted Value')
plot_compare('Clogging Index 1','Clogging Index','Datapoints',range(len(actual[10000:11000])),actual[10000:11000],prediction[10000:11000],'g','r','Actual Value','Predicted Value')
plot_compare('Clogging Index 1','Clogging Index','Datapoints',range(len(actual[9000:10000])),actual[9000:10000],prediction[9000:10000],'g','r','Actual Value','Predicted Value')

plot_compare('Training Loss vs. Validataion Loss (Clogging Index 1)','Mean Square Error','Epoch',range(len(loss_train)),
            loss_train,loss_validation,'g','r','Training Loss','Validation Loss')
plt.plot(values_2)
training_pred = sc_y_1.inverse_transform(model_final.predict(X_train_1))
plot_compare('Clogging Index 1','Clogging Index','Datapoints',range(len(training_pred[6500:6800])),training_pred[6500:6800],sc_y_1.inverse_transform(y_train_1[6500:6800]),'g','r','Actual Value','Predicted Value')

#Save the results into two excel files
dic_1 = {'Training loss 1':loss_train,'Validation loss 1':loss_validation,}

df_loss = pd.DataFrame.from_dict(dic_1)
df_pred = pd.DataFrame(prediction,columns=['Prediction'])
df_actual = pd.DataFrame(actual,columns=['Actual'])

#Remember to change the grade in file name
df_loss.to_csv('Training vs Validation loss C1 (Ca).csv')
df_pred.to_csv('Prediction C1 (Ca).csv')
df_actual.to_csv('Actual C1 (Ca).csv')
