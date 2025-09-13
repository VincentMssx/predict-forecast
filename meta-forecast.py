import sys
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

################################################################################
# 1. Get Data (Using a longer, more robust date range for better training)
################################################################################

# --- Configuration ---
LATITUDE = 46.244
LONGITUDE = -1.561
START_DATE = "2025-09-01"
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


################################################################################
# 2. Data Preparation and Vector Conversion
################################################################################

# *** THIS IS THE CORRECTED, MORE ROBUST to_vector FUNCTION ***
def to_vector(df):
    direction_cols = [col for col in df.columns if 'winddirection' in col]
    for dir_col in direction_cols:
        speed_col = dir_col.replace('winddirection', 'windspeed')
        prefix = dir_col.replace('winddirection', '')
        if speed_col in df.columns:
            speed = df[speed_col]
            direction_rad = np.deg2rad(df[dir_col])
            df[f'{prefix}u'] = -speed * np.sin(direction_rad)
            df[f'{prefix}v'] = -speed * np.cos(direction_rad)
    return df

df_vectors = to_vector(final_df.copy())
print(df_vectors.describe())
print("\nVector columns created successfully.")


################################################################################
# 3. Model Evaluation (Overall and by Direction)
################################################################################

print("\n--- Evaluating Models with Scikit-learn (Overall Performance) ---")
y_true_speed = df_vectors['gt_windspeed']
y_true_u = df_vectors['gt_u']
y_true_v = df_vectors['gt_v']
results = {}

for model in forecast_models:
    y_pred_speed = df_vectors[f'{model}_windspeed']
    mae_speed = mean_absolute_error(y_true_speed, y_pred_speed)
    rmse_speed = np.sqrt(mean_squared_error(y_true_speed, y_pred_speed))
    
    y_pred_u = df_vectors[f'{model}_u']
    y_pred_v = df_vectors[f'{model}_v']
    direction_error = np.sqrt((y_true_u - y_pred_u)**2 + (y_true_v - y_pred_v)**2)
    mae_direction = direction_error.mean()
    
    results[model] = {'MAE_Speed': mae_speed, 'RMSE_Speed': rmse_speed, 'MAE_Direction_Vector': mae_direction}

results_df = pd.DataFrame(results).T
print(results_df.sort_values(by='RMSE_Speed'))

# ### NEW SECTION: METHOD 1 - ANALYZE PERFORMANCE BY WIND DIRECTION ###
print("\n--- Evaluating Models by Wind Direction Sector ---")

def get_wind_sector(deg):
    if 337.5 <= deg <= 360 or 0 <= deg < 22.5: return 'N'
    if 22.5 <= deg < 67.5: return 'NE'
    if 67.5 <= deg < 112.5: return 'E'
    if 112.5 <= deg < 157.5: return 'SE'
    if 157.5 <= deg < 202.5: return 'S'
    if 202.5 <= deg < 247.5: return 'SW'
    if 247.5 <= deg < 292.5: return 'W'
    if 292.5 <= deg < 337.5: return 'NW'

df_vectors['gt_sector'] = df_vectors['gt_winddirection'].apply(get_wind_sector)
sectors = sorted(df_vectors['gt_sector'].unique())
sector_results = []

for sector in sectors:
    sector_df = df_vectors[df_vectors['gt_sector'] == sector]
    if len(sector_df) < 20: continue # Skip sectors with too little data

    y_true_speed_sector = sector_df['gt_windspeed']
    
    for model in forecast_models:
        y_pred_speed_sector = sector_df[f'{model}_windspeed']
        rmse_speed_sector = np.sqrt(mean_squared_error(y_true_speed_sector, y_pred_speed_sector))
        sector_results.append({'Sector': sector, 'Model': model, 'RMSE_Speed': rmse_speed_sector})

sector_results_df = pd.DataFrame(sector_results)
print("\nWind Speed RMSE by Sector (Lower is better):")
# Pivot for easy comparison
print(sector_results_df.pivot(index='Sector', columns='Model', values='RMSE_Speed'))


################################################################################
# 4. Meta-Forecast with PyTorch (IMPROVED MODEL)
################################################################################

print("\n--- Building an Improved Corrective Model with PyTorch ---")

# ### NEW SECTION: METHOD 2 - FEATURE ENGINEERING FOR PYTORCH MODEL ###
# We will use AROME's predicted direction to create cyclical features.
# AROME is often the higher-resolution model, making it a good choice.
best_model_for_features = 'arome_france' 
arome_dir_rad = np.deg2rad(df_vectors[f'{best_model_for_features}_winddirection'])
df_vectors['arome_dir_sin'] = np.sin(arome_dir_rad)
df_vectors['arome_dir_cos'] = np.cos(arome_dir_rad)

print("Added 'arome_dir_sin' and 'arome_dir_cos' as new features.")

# 1. Prepare Data with the new features
features_cols = ([f'{model}_windspeed' for model in forecast_models] +
                 [f'{model}_u' for model in forecast_models] +
                 [f'{model}_v' for model in forecast_models] +
                 ['arome_dir_sin', 'arome_dir_cos']) # Add the new features here

features = df_vectors[features_cols].values
targets = df_vectors[['gt_windspeed', 'gt_u', 'gt_v']].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 2. Define the Neural Network
class WindNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(WindNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

# 3. Training the Model
input_size = X_train.shape[1] # This will now be larger due to new features
output_size = y_train.shape[1]
model = WindNet(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 4. Evaluate the PyTorch Model
with torch.no_grad():
    model.eval()
    y_pred_pytorch = model(X_test_tensor).numpy()

mae_speed_pytorch = mean_absolute_error(y_test[:, 0], y_pred_pytorch[:, 0])
rmse_speed_pytorch = np.sqrt(mean_squared_error(y_test[:, 0], y_pred_pytorch[:, 0]))
direction_error_pytorch = np.sqrt((y_test[:, 1] - y_pred_pytorch[:, 1])**2 + (y_test[:, 2] - y_pred_pytorch[:, 2])**2)
mae_direction_pytorch = direction_error_pytorch.mean()

results['PyTorch_Ensemble_Improved'] = {
    'MAE_Speed': mae_speed_pytorch,
    'RMSE_Speed': rmse_speed_pytorch,
    'MAE_Direction_Vector': mae_direction_pytorch
}

# Display final comparison
final_results_df = pd.DataFrame(results).T
print("\n--- Final Comparison Including Improved PyTorch Model ---")
print(final_results_df.sort_values(by='RMSE_Speed'))