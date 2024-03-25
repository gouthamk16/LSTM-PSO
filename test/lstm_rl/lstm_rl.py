import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Download stock data
symbol = 'CSCO'
start_date = '2018-01-01'
end_date = '2024-01-01'
stock_data = yf.download(symbol, start=start_date, end=end_date)
specific_df = pd.DataFrame(stock_data).reset_index()

# Preprocess stock data
specific_df['Date'] = pd.to_datetime(specific_df['Date'])
specific_df['Year'] = specific_df['Date'].dt.year
specific_df['Month'] = specific_df['Date'].dt.month
specific_df['Day'] = specific_df['Date'].dt.day

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(specific_df[['Close']])

# Split data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

# Generate sequences and labels for LSTM
def generate_sequences_and_labels(data, n_past):
    sequences, labels = [], []
    for i in range(len(data) - n_past):
        sequences.append(data[i:i + n_past])
        labels.append(data[i + n_past, 0])
    return np.array(sequences), np.array(labels)

n_past = 30
x_train, y_train = generate_sequences_and_labels(train_data, n_past)
x_test, y_test = generate_sequences_and_labels(test_data, n_past)
x_train_lstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test_lstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

def objective_function(params, X_train, Y_train, X_test, Y_test):

    lstm_units, dropout_rate, dense_units = params

    model = Sequential()
    model.add(LSTM(units=int(lstm_units), return_sequences=True, input_shape=(n_past, 1)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=int(lstm_units/2), return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(int(dense_units), activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=50, verbose=1)

    predictions = model.predict(X_test)
    r2 = r2_score(Y_test, predictions)
    # mse = mean_squared_error(Y_test, predictions)
    return -r2 # negative R-squared for minimization

def particle_swarm_optimization(objective_function, no_particle, no_dim, x_range, v_range, iw_range, c, X_train, Y_train, X_test, Y_test, print_step, iter):

    particles = np.random.uniform([r[0] for r in x_range], [r[1] for r in x_range], size=(no_particle, no_dim))
    velocities = np.random.uniform([r[0] for r in v_range], [r[1] for r in v_range], size=(no_particle, no_dim))
    pbest = np.full(no_particle, np.inf)
    pbestpos = np.zeros((no_particle, no_dim))
    gbest = np.inf
    gbestpos = np.zeros((no_dim,))

    for i in range(iter):

        for j, particle in enumerate(particles):

            fitness = objective_function(particle, X_train, Y_train, X_test, Y_test)

            if fitness < pbest[j]:
                pbest[j] = fitness
                pbestpos[j] = particle.copy()
            if fitness < gbest:
                gbest = fitness
                gbestpos = particle.copy()

        for j, particle in enumerate(particles):
            iw = np.random.uniform(iw_range[0], iw_range[1], 1)[0]
            velocities[j] = (
                iw * velocities[j]
                + (c[0] * np.random.uniform(0.0, 1.0, (no_dim,)) * (pbestpos[j] - particle))
                + (c[1] * np.random.uniform(0.0, 1.0, (no_dim,)) * (gbestpos - particle))
            )
            particles[j] += velocities[j]

    return gbestpos

# Run PSO optimization
no_particle = 10
no_dim = 3
x_range = [(50, 200), (0.1, 0.5), (8, 32)]
v_range = [(1, 10), (0.01, 0.1), (1, 10)]
iw_range = (0.1, 0.5)
c = (1.4962, 1.4962)  # cognitive and social parameters
# Run PSO optimization
best_params = particle_swarm_optimization(objective_function, no_particle, no_dim, x_range, v_range, iw_range, c, x_train, y_train, x_test, y_test, 10, 3)

# Define the ranges for LSTM parameters
lstm_units, dropout_rate, dense_units = best_params

# Define the create the main lstm model for final prediction

def lstm_Model(lstm_units, dropout_rate, dense_units, n_past):

  lstm_model = Sequential()
  lstm_model.add(LSTM(units=int(lstm_units), return_sequences=True, input_shape=(n_past, 1)))
  lstm_model.add(Dropout(dropout_rate))
  lstm_model.add(LSTM(units=int(lstm_units/2), return_sequences=False))
  lstm_model.add(Dropout(dropout_rate))
  lstm_model.add(Dense(int(dense_units), activation='relu'))
  lstm_model.add(Dense(1))
  lstm_model.compile(optimizer='adam', loss='mean_squared_error')

  return lstm_model

# Train the LSTM model
lstm_model = lstm_Model(lstm_units, dropout_rate, dense_units, n_past)
history = lstm_model.fit(x_train_lstm, y_train, epochs=200, validation_data=(x_test_lstm, y_test), verbose=1)

# Plotting the learning curve

predictions = lstm_model.predict(x_test)
# Inverse transform the predicted and actual prices
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))



# Calculate evaluation metrics
r2 = r2_score(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)
mse = mean_squared_error(actual_prices, predicted_prices)
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

# Print evaluation metrics
print("R-squared:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("MAPE:", mape)

# Plot actual vs predicted prices
plt.plot(actual_prices, label='Actual')
plt.plot(predicted_prices, label='Predicted')
plt.title('Stock Price Prediction using LSTM+PSO')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()