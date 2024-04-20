import numpy as np
import pandas as pd
import math
import os
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import yfinance as yf
import tensorflow as tf

import csv


def stock_predict_1D(stock_name, filename):
    # Downloading stock data from Yahoo Finance
    stock_data = yf.download(stock_name, start='2019-01-01', end='2024-01-01')
    specific_df = pd.DataFrame(stock_data).reset_index()
    specific_df['Name'] = stock_name

    new_df = specific_df.reset_index()['Close']
    scaler=MinMaxScaler()
    scaled_data = scaler.fit_transform(np.array(new_df).reshape(-1,1))

    from sklearn.model_selection import train_test_split
    test_size = 0.2
    train_data, test_data = train_test_split(scaled_data, test_size=test_size, shuffle=False)

    def generate_sequences_and_labels(data, n_past):
        sequences = [data[i - n_past:i, 0] for i in range(n_past, len(data))]
        labels = [data[i, 0] for i in range(n_past, len(data))]
        return np.array(sequences), np.array(labels)

    n_past = 60
    x_train, y_train = generate_sequences_and_labels(train_data, n_past)
    x_test, y_test = generate_sequences_and_labels(test_data, n_past)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)


    filename_a = filename + "_lstm1d.keras"
    model = tf.keras.models.load_model(filename_a)

    def make_predictions(model, x_train, x_test):
        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)
        return train_predict, test_predict

    train_predict, test_predict = make_predictions(model, x_train, x_test)

    def inverse_transform(scaler, y_train, train_predict, y_test, test_predict):
        y_test = y_test.reshape(-1,1)
        y_train = y_train.reshape(-1,1)
        y_test = scaler.inverse_transform(y_test)
        y_train = scaler.inverse_transform(y_train)
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        return y_test, y_train, train_predict, test_predict

    y_test, y_train, train_predict, test_predict = inverse_transform(scaler, y_train, train_predict, y_test, test_predict)

    # def evaluate_model(y_train, train_predict, y_test, test_predict):
    #     #train_mse = mean_squared_error(y_train, train_predict)
    #     test_mse = mean_squared_error(y_test, test_predict)
    #     test_mae = mean_absolute_error(y_test, test_predict)
    #     test_mape = mean_absolute_percentage_error(y_test, test_predict)
    #     test_rs = r2_score(y_test, test_predict)
    #     #print(f"Training MSE: {train_mse}")
    #     print(f"MSE of {stock_name}: {test_mse}")
    #     print(f"MAE of {stock_name}: {test_mae}")
    #     print(f"MAPE of {stock_name}: {test_mape}")
    #     print(f"R Sqaured of {stock_name}: {test_rs}")

    # evaluate_model(y_train, train_predict, y_test, test_predict)

    # Calculate evaluation metrics
    test_mse = mean_squared_error(y_test, test_predict)
    test_mae = mean_absolute_error(y_test, test_predict)
    test_mape = mean_absolute_percentage_error(y_test, test_predict)
    test_rs = r2_score(y_test, test_predict)
    
    return [test_mae,test_mse,test_mape,test_rs]

stock_list = ["AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW","GS","HD","HON",
              "IBM","INTC","JNJ","JPM","KO","MCD","MMM","MRK","MSFT","NKE","PG","TRV","UNH",
              "V","VZ","WBA","WMT"]

with open("mae_metrics.csv",'r') as file:
    r = csv.reader(file)
    l_mae = []
    for i in r:
        l_mae.append(i)

with open("mse_metrics.csv",'r') as file:
    r = csv.reader(file)
    l_mse = []
    for i in r:
        l_mse.append(i)

with open("mape_metrics.csv",'r') as file:
    r = csv.reader(file)
    l_mape = []
    for i in r:
        l_mape.append(i)

with open("r2_metrics.csv",'r') as file:
    r = csv.reader(file)
    l_r2 = []
    for i in r:
        l_r2.append(i)

l_mae[0].append("LSTM1D")
l_mse[0].append("LSTM_1D")
l_mape[0].append("LSTM_1D")
l_r2[0].append("LSTM_1D")

# for i in range(len(stock_list)):
#     result = stock_predict_1D(stock_list[i],stock_list[i])
#     l[i+1].append(result)


for i in range(len(stock_list)):
    result = stock_predict_1D(stock_list[i],stock_list[i])
    l_mae[i+1].append(result[0])
    l_mse[i+1].append(result[1])
    l_mape[i+1].append(result[2])
    l_r2[i+1].append(result[3])

with open('mae_metrics.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerows(l_mae)

with open('mse_metrics.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerows(l_mse)

with open('mape_metrics.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerows(l_mape)

with open('r2_metrics.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerows(l_r2)