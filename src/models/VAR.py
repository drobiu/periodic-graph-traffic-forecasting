from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Load dataset
data = pd.read_csv('../../data/METR-LA_clean.csv', sep=",", header=None)
print(data.shape)

window_size = 24  # the number of points we use for training
forecast_horizon = 12  # the number of points we want to predict

mae_errors_15min = []
rmse_errors_15min = []
mape_errors_15min = []

mae_errors_30min = []
rmse_errors_30min = []
mape_errors_30min = []

mae_errors_60min = []
rmse_errors_60min = []
mape_errors_60min = []

# convert data to numpy array
data = data.to_numpy()

data = data.T
print(data.shape)

for i in range(window_size, len(data)-forecast_horizon):
    model = VAR(data[i-window_size:i])
    model_fit = model.fit()
    
    forecast = model_fit.forecast(model_fit.y, steps=forecast_horizon)

    for j in range(forecast_horizon):
        mae = mean_absolute_error(data[i+j], forecast[j])
        rmse = np.sqrt(mean_squared_error(data[i+j], forecast[j]))
        mape = np.mean(np.abs((data[i+j] - forecast[j]) / data[i+j])) * 100

        if j == 2:  # 15 minutes ahead
            mae_errors_15min.append(mae)
            rmse_errors_15min.append(rmse)
            mape_errors_15min.append(mape)
        elif j == 5:  # 30 minutes ahead
            mae_errors_30min.append(mae)
            rmse_errors_30min.append(rmse)
            mape_errors_30min.append(mape)
        elif j == 11:  # 60 minutes ahead
            mae_errors_60min.append(mae)
            rmse_errors_60min.append(rmse)
            mape_errors_60min.append(mape)

print(f"MAE 15min: {np.mean(mae_errors_15min)}")
print(f"RMSE 15min: {np.mean(rmse_errors_15min)}")
print(f"MAPE 15min: {np.mean(mape_errors_15min)}")

print(f"MAE 30min: {np.mean(mae_errors_30min)}")
print(f"RMSE 30min: {np.mean(rmse_errors_30min)}")
print(f"MAPE 30min: {np.mean(mape_errors_30min)}")

print(f"MAE 60min: {np.mean(mae_errors_60min)}")
print(f"RMSE 60min: {np.mean(rmse_errors_60min)}")
print(f"MAPE 60min: {np.mean(mape_errors_60min)}")
