import pandas as pd
import numpy as np
import requests
from pandas import (DataFrame, HDFStore)

# Google Drive links
metr_la_link = "https://drive.google.com/file/d/1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC/view?usp=drive_link"

# Extract file ID from the link
metr_la_id = metr_la_link.split("/")[5]

# Construct download URL
metr_la_download_url = f"https://drive.google.com/uc?id={metr_la_id}"

# Download and save the file
metr_la_response = requests.get(metr_la_download_url)
metr_la_output_path = "METR-LA/metr-la.h5" 
with open(metr_la_output_path, "wb") as file:
    file.write(metr_la_response.content)
print("Metr-la.h5 file downloaded successfully.")

# Convert the file to CSV
metr_la_csv_output_path = "METR-LA/metr-la.csv"
metr_la = HDFStore(metr_la_output_path)

# Convert the data frame to CSV
metr_la_key = metr_la.keys()[0]
metr_la_df = metr_la[metr_la_key]
metr_la_df.to_csv(metr_la_csv_output_path, index=False)
metr_la.close()
print("Metr-la.csv file created successfully.")

# Create clean CSV file, removing the first row and the first column
metr_la_clean_csv_output_path = "METR-LA/metr-la_clean.csv"
metr_la_df = pd.read_csv(metr_la_csv_output_path)
metr_la_df = metr_la_df.iloc[1:, 1:]
metr_la_df.to_csv(metr_la_clean_csv_output_path, index=False)
print("Metr-la_clean.csv file created successfully.")

# Convert the dataframe to a numpy array and save it to a npz file
data_array = metr_la_df.values
np.savez('METR-LA/metr-la.npz', data=data_array)

# Create a smaller CSV file for testing
metr_la_small_csv_output_path = "METR-LA/metr-la_small.csv"
metr_la_df = pd.read_csv(metr_la_clean_csv_output_path)
metr_la_df = metr_la_df.iloc[:1000, :50]
metr_la_df.to_csv(metr_la_small_csv_output_path, index=False)
print("Metr-la_small.csv file created successfully.")

# Convert the dataframe to a numpy array and save it to a npz file
data_array = metr_la_df.values
np.savez('METR-LA/metr-la_small.npz', data=data_array)
