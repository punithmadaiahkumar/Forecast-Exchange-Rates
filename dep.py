# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 17:45:47 2021

@author: Punith Gowda
"""
#Library
#VIZ AND DATA MANIPULATION LIBRARY
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#import math
#numpy
#from numpy import array

#warnings
import warnings 
warnings.filterwarnings('ignore')

#Library
from sklearn.preprocessing import MinMaxScaler

#Defining the LSTM model
from keras.models import Sequential
from keras.layers import Dense,LSTM

#METRICS
#from sklearn.metrics import mean_squared_error

#Streamlit
import streamlit as st

#Pickle
#import pickle
#from pickle import load


#Title
st.title(" Forecast Exchange Rates ")
st.write("## USD to INR")



inrusd = pd.read_csv('Dataset.csv',parse_dates=["observation_date"])
df = inrusd.copy()
#renaming the date and rate
data = df[['observation_date', 'DEXINUS']]
data.columns = ['date', 'rate']
#numeric
data['rate'] = pd.to_numeric(data.rate)
#Sort to ascending
data = data.sort_values('date', ascending=True)
#Forward Filling
data.fillna(method='ffill', inplace=True)
#transformation of values to float
data['rate'] = pd.to_numeric(data['rate'], downcast="float")

#spliting data for lstm
data1 = data.iloc[10000:]
data2 = data1.copy() 
#reset index of rate
df1=data2.reset_index()['rate']

#MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

#Splitting
#splitting dataset into train and test split
training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

# convertsion of an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

#LSTM MODEL
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

#model fit
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=5,batch_size=64,verbose=1)

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

#Length of test data
len_test_data = len(test_data)

#subtracting 100 data from test data
look_back=100
x_input=test_data[(len_test_data - look_back):].reshape(1,-1)

#temp list
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

#future prediction
#INPUT
obv_date = st.text_input("Enter your number of days here")
obv_date=float(int(obv_date))
#obv_date = 30
lst_output=[]
n_steps=100
i=0
while(i<obv_date):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
 
    
day_new=np.arange(1,101)
day_pred=np.arange(101,(101 + obv_date))

#original length
len_val = len(df1)

#prediction values
op_res = scaler.inverse_transform(lst_output)
#print(op_res)
op_res1 = pd.DataFrame(op_res, columns=['Prediction'])
st.write(op_res1)




















