import sys

sys.path.append('lstm_1d') 
import lstm_1d as LSTM1D

# sys.path.append('lstm_2d')
# import lstm_2d as LSTM2D

# sys.path.append('lstm_3d')
# import lstm_3d as LSTM3D

# sys.path.append('lstm_ga')
# import lstm_ga as LSTMGA

# sys.path.append('ann')
# import ann as ANN


stock_list = ["AXP","AMGN","AAPL",]

for i in stock_list:
    filename = i
    LSTM1D.stock_predict_1D(i, filename)
    # LSTM2D.stock_predict_2D(i, filename)
    # LSTM3D.stock_predict_3D(i, filename)
    # LSTMGA.stock_predict_ANN(i, filename)
    # ANN.stock_predict_GA(i, filename)
