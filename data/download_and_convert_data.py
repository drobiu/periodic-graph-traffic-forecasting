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
metr_la_output_path = "metr-la.h5" 
with open(metr_la_output_path, "wb") as file:
    file.write(metr_la_response.content)
print("Metr-la.h5 file downloaded successfully.")

# Convert the file to CSV
metr_la_csv_output_path = "metr-la.csv"
metr_la = HDFStore(metr_la_output_path)

# Convert the data frame to CSV
metr_la_key = metr_la.keys()[0]
metr_la_df = metr_la[metr_la_key]
metr_la_df.to_csv(metr_la_csv_output_path, index=False)
metr_la.close()
print("Metr-la.csv file created successfully.")




