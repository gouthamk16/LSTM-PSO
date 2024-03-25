import numpy as np
import pandas as pd
import math
import os
import random
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import yfinance as yf
import tensorflow as tf


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

    ##########################START############################################
    # def build_lstm_model(input_shape):
    #     model = Sequential()
    #     model.add(LSTM(units=50, input_shape=[x_train.shape[1], 1]))
    #     model.add(Dense(10))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(1))
    #     return model
    #
    # def compile_lstm_model(model):
    #     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])
    #
    #
    # def train_lstm_model(model, x_train, y_train, x_test, y_test):
    #     model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=250, batch_size=32, verbose=1)
    #
    #
    # model = build_lstm_model(x_train.shape[1])
    # compile_lstm_model(model)
    # train_lstm_model(model, x_train, y_train, x_test, y_test)
    #############################################END#################################################

    folder_path = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
    filename_a = os.path.join(folder_path, filename + "1D.keras")
    # model.save(filename_a)

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

    def evaluate_model(y_train, train_predict, y_test, test_predict):
        #train_mse = mean_squared_error(y_train, train_predict)
        test_mse = mean_squared_error(y_test, test_predict)
        test_mae = mean_absolute_error(y_test, test_predict)
        test_mape = mean_absolute_percentage_error(y_test, test_predict)
        test_rs = r2_score(y_test, test_predict)
        #print(f"Training MSE: {train_mse}")
        print(f"MSE of {stock_name}: {test_mse}")
        print(f"MAE of {stock_name}: {test_mae}")
        print(f"MAPE of {stock_name}: {test_mape}")
        print(f"R Sqaured of {stock_name}: {test_rs}")

    evaluate_model(y_train, train_predict, y_test, test_predict)


    predictions = model.predict(x_test)

    actual_values_inverse = y_test
    predicted_values_inverse = scaler.inverse_transform(predictions)

    # comparison_df = pd.DataFrame({'Actual Close Prices': actual_values_inverse.flatten(), 'Predicted Close Prices': predicted_values_inverse.flatten()})
    # print(comparison_df.head())

    plt.figure(figsize=(10, 5))
    plt.plot(actual_values_inverse.flatten(), label="Actual Close Prices")
    plt.plot(predicted_values_inverse.flatten(), 'r', label="Predicted Close Prices")
    plt.ylabel('Close Price')
    plt.xlabel('Time Step')
    plt.legend()
    plt.title(stock_name+'-LSTM 1D')
    # plt.show()

    plt_name = os.path.join(folder_path, f"{stock_name}1D.png")
    # plt.savefig(plt_name)