from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np
from tqdm import tqdm

# Load dataset
data = pd.read_csv('../../data/METR-LA_clean.csv', sep=",", header=None)
# Convert data to numpy array
data = data.to_numpy()
# Transpose data to have sensors as rows
data = data.T
# Create a smaller dataset
data_small = data[:50, :500]

# Look at 10 random nodes to average their values
random_nodes = np.random.randint(0, 50, 10)

# Time horizons of interest are 15, 30 and 60 minutes
time_horizons = [15, 30, 60]
timesteps = [int(round(x / 5)) for x in time_horizons]

mae_per_horizon = []
rmse_per_horizon = []
mape_per_horizon = []

for timestep_horizon in timesteps:
    print("Time horizon: ", timestep_horizon*5, " minutes")
    mae_list = []
    rmse_list = []
    mape_list = []

    #for node_idx in random_nodes:
    for node_idx in tqdm(random_nodes, desc="Processing nodes", unit="node"):
        node = data_small[node_idx]
        # split into train and test sets
        size = int(len(node) * 0.66)
        train, test = node[0:size], node[size:len(node)]
        history = [x for x in train]
        predictions = list()
        # walk-forward validation
        for t in range(len(test)-12): 
            model = ARIMA(history, order=(12,1,0))
            model_fit = model.fit()
            # Forecast t steps ahead and take the last prediction as the forecasted value
            output = model_fit.forecast(steps=timestep_horizon)
            yhat = output[-1] 
            predictions.append(yhat)
            obs = test[t+12] 
            history.append(obs)
            #print('predicted=%f, expected=%f' % (yhat, obs))
        # evaluate forecasts, avoid the last 12 values in test to match the lengths
        mae = mean_absolute_error(test[:-12], predictions)
        rmse = sqrt(mean_squared_error(test[:-12], predictions))
        mape = np.mean(np.abs((test[:-12] - predictions) / (test[:-12]+1e-10))) * 100  
        # print('Test MAE: %.3f' % mae)      
        # print('Test RMSE: %.3f' % rmse)
        # print('Test MAPE: %.3f' % mape)
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
        # plot forecasts against ground truth
        # Avoid the last 12 values in test to match the lengths
        #plt.plot(test[:-12]) 
        #plt.plot(predictions, color='red')
        #plt.show()

    Average_MAE = np.mean(mae_list)
    Average_RMSE = np.mean(rmse_list)
    Average_MAPE = np.mean(mape_list)

    mae_per_horizon.append(Average_MAE)
    rmse_per_horizon.append(Average_RMSE)
    mape_per_horizon.append(Average_MAPE)

# Print the average errors for each time horizon
print("Average MAE per time horizon: ", mae_per_horizon)
print("Average RMSE per time horizon: ", rmse_per_horizon)
print("Average MAPE per time horizon: ", mape_per_horizon)
