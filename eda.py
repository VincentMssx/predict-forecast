import sys
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

################################################################################
# 1. Get Data
################################################################################

# --- Configuration ---
LATITUDE = 46.244
LONGITUDE = -1.561
START_DATE = "2025-06-12"
END_DATE = "2025-09-13"

# --- Function to Fetch Data from Open-Meteo ---
def fetch_weather_data(api_url, params):
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        hourly_data = data['hourly']
        df = pd.DataFrame(hourly_data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        if 'response' in locals() and response.text:
            print(f"API Response: {response.json().get('reason', 'No reason provided')}")
        return None
    except KeyError:
        print("Error: 'hourly' key not found in API response.")
        return None

# --- Fetch Ground Truth Data ---
print("Fetching ground truth data (ERA5)...")
historical_api_url = "https://archive-api.open-meteo.com/v1/archive"
params_ground_truth = {
    "latitude": LATITUDE, "longitude": LONGITUDE, "start_date": START_DATE,
    "end_date": END_DATE, "hourly": "windspeed_10m,winddirection_10m"
}
ground_truth_df = fetch_weather_data(historical_api_url, params_ground_truth)
ground_truth_df.rename(columns={'windspeed_10m': 'gt_windspeed', 'winddirection_10m': 'gt_winddirection'}, inplace=True)

# --- Fetch Historical Forecast Data ---
print("Fetching forecast data...")
forecast_api_url = "https://api.open-meteo.com/v1/forecast"
forecast_models = ["arome_france", "gfs_seamless"]
forecast_dfs = {}
for model in forecast_models:
    print(f"  - Fetching {model}...")
    params_forecast = {
        "latitude": LATITUDE, "longitude": LONGITUDE, "start_date": START_DATE,
        "end_date": END_DATE, "hourly": "windspeed_10m,winddirection_10m", "models": model
    }
    df = fetch_weather_data(forecast_api_url, params_forecast)
    if df is not None:
        df.rename(columns={
            'windspeed_10m': f'{model}_windspeed',
            'winddirection_10m': f'{model}_winddirection'
        }, inplace=True)
        forecast_dfs[model] = df


# --- Combine Data ---
print("Combining all data sources...")
if ground_truth_df is not None and forecast_dfs:
    final_df = ground_truth_df
    for model_name, forecast_df in forecast_dfs.items():
        final_df = final_df.join(forecast_df, how='inner')
    final_df.dropna(inplace=True)
    if final_df.empty:
        print("No overlapping data found. Exiting.")
        exit()
    print("Data successfully downloaded and combined.")
else:
    print("Failed to download or combine data. Exiting.")
    exit()


# --- Exploratory Data Analysis ---

print("\n--- Starting Exploratory Data Analysis ---")

# 2. Distribution of Wind Speed (Histograms)
plt.figure(figsize=(12, 6))
sns.histplot(final_df['gt_windspeed'], color='black', label='Ground Truth', kde=True, stat="density")
sns.histplot(final_df['arome_france_windspeed'], color='blue', label='AROME', kde=True, stat="density")
sns.histplot(final_df['gfs_seamless_windspeed'], color='green', label='GFS', kde=True, stat="density")
plt.title('Distribution of Wind Speeds')
plt.xlabel('Wind Speed (km/h)')
plt.legend()

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
print(final_df.columns)
ax.hist(np.deg2rad(final_df['gt_winddirection']), bins=36, alpha=0.7, label='Ground Truth')
ax.hist(np.deg2rad(final_df['arome_france_winddirection']), bins=36, alpha=0.3, label='AROME')
plt.title('Wind Direction Distribution (Ground Truth vs. AROME vs. GFS)')
plt.legend()

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
print(final_df.columns)
ax.hist(np.deg2rad(final_df['gt_winddirection']), bins=36, alpha=0.7, label='Ground Truth')
ax.hist(np.deg2rad(final_df['gfs_seamless_winddirection']), bins=36, alpha=0.3, label='GFS')
plt.title('Wind Direction Distribution (Ground Truth vs. AROME vs. GFS)')
plt.legend()
plt.show()

