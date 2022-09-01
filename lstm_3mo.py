# -*- coding: utf-8 -*-

!apt-get install openjdk-8-jdk-headless -qq > /dev/null

!wget -q https://dlcdn.apache.org/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz

!tar xf spark-3.2.1-bin-hadoop3.2.tgz

!pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.2.1-bin-hadoop3.2"

from google.colab import drive
drive.mount('/content/drive')

import findspark
findspark.init()

import pyspark
import random

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout
import matplotlib.pyplot as plt

fileName = "/content/drive/MyDrive/6889proj/Dataset/aapl_history_3mo.csv"

#initial a spark cluster
spark = pyspark.sql.SparkSession.builder.appName("StockPrice").config("spark.some.config.option", "some-value").getOrCreate()
#spark = pyspark.sql.SparkSession.builder.master("local").appName("StockMarket").config("spark.executor.memory", "6gb").getOrCreate()
#Use spark.read() to access this. Loads a CSV file and returns the result as a DataFrame.
df = spark.read.csv(fileName, header=True, inferSchema=True)

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
data = df.select('Open','High','Low','Volume','SentimentScore','Close').toPandas()
print(data)
# data['Close'] = data['Close'].shift(-5,fill_value= 0)
# data.drop(data.tail(5).index,inplace=True)
# print(data)
dataset = data.to_numpy()

#Get /Compute the number of rows to train the model on
training_data_len = math.ceil(len(dataset) *.8) 

#Create the scaled training data set and test data set
train_data = dataset[0:training_data_len]
test_data = dataset[training_data_len :]

print(train_data[0], len(train_data))
print('-------------')
print(test_data[0], len(test_data))


#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.fit_transform(test_data)

print(scaled_train_data[0])
print('-------------')
print(scaled_test_data[0])

xtrain = scaled_train_data[:, 0:-1]
ytrain = scaled_train_data[:, -1:]

xtest = scaled_test_data[:, 0:-1]
ytest = scaled_test_data[:, -1:]

print('xtrain shape = {}'.format(xtrain.shape))
print('xtest shape = {}'.format(xtest.shape))
print('ytrain shape = {}'.format(ytrain.shape))
print('ytest shape = {}'.format(ytest.shape))

plt.figure(figsize=(16,6))
plt.plot(xtrain[:,0],color='red', label='open')
plt.plot(xtrain[:,1],color='blue', label='high')
plt.plot(xtrain[:,2],color='green', label='low')
plt.legend(loc = 'upper left')
plt.title('Open, High, Low by Day')
plt.xlabel('Days')
plt.ylabel('Scaled Quotes')
plt.show()

#volume and Polarity are very noisy
plt.figure(figsize=(16,6))
plt.plot(xtrain[:,0],color='red', label='open')
plt.plot(xtrain[:,1],color='blue', label='high')
plt.plot(xtrain[:,2],color='green', label='low')
plt.plot(xtrain[:,3],color='yellow', label='volume')
plt.plot(xtrain[:,4],color='purple', label='Polarity')
plt.legend(loc = 'upper left')
plt.title('After add volume and Sentiment Score')
plt.xlabel('Days')
plt.ylabel('Scaled Quotes')
plt.show()

from keras import models, layers
model = models.Sequential()
model.add(layers.LSTM(10, input_shape=(1,5)))
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(xtrain.shape)
xtrain = xtrain.reshape((xtrain.shape[0], 1, xtrain.shape[1]))
print(xtrain)
xtest  = xtest.reshape((xtest.shape[0], 1, xtest.shape[1]))
print('The shape of xtrain is {}: '.format(xtrain.shape))
print('The shape of xtest is {}: '.format(xtest.shape))

loss = model.fit(xtrain, ytrain, batch_size=20, epochs=50)

plt.plot(loss.history['loss'], label = 'loss')
plt.title('mean squared error by epoch')
plt.legend()
plt.show()

#Getting the models predicted price values
predictions = model.predict(xtest) 
print(predictions)

#Undo scaling, show real prediction
# create empty table with 2 fields
trainPredict_dataset_like = np.zeros(shape=(len(predictions), 6) )
# put the predicted values in the right field
trainPredict_dataset_like[:,0] = predictions[:,0]
# inverse transform and then select the right field
predictions = scaler.inverse_transform(trainPredict_dataset_like)[:,0]


train = data.loc[:training_data_len]
valid = data.loc[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# test_dataset_like = np.zeros(shape=(len(predictions), 6) )
# test_dataset_like[:,0] = valid['Close'][:,0]

# scaler = MinMaxScaler(feature_range=(0, 1)) 
# ScaledValidation= scaler.fit_transform(valid['Close'])

rmse=np.sqrt(np.mean(((trainPredict_dataset_like- ytest)**2)))
rmse