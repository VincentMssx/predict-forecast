
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

############################
# 1. Get Data
############################

# --- Configuration ---
LATITUDE = 46.244
LONGITUDE = -1.561
START_DATE = "2025-09-01"
END_DATE = "2025-09-13"

# --- Function to Fetch Data from Open-Meteo ---
def fetch_weather_data(api_url, params):
    """Fetches data from the Open-Meteo API and returns a pandas DataFrame."""
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        # print(data)
        
        hourly_data = data['hourly']
        df = pd.DataFrame(hourly_data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except KeyError:
        print("Error: 'hourly' key not found in API response. Check your parameters.")
        return None

# --- 1. Fetch Ground Truth (Historical Reanalysis) Data ---
print("Fetching ground truth data (ERA5)...")
historical_api_url = "https://archive-api.open-meteo.com/v1/archive"
params_ground_truth = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": "windspeed_10m,winddirection_10m"
}
ground_truth_df = fetch_weather_data(historical_api_url, params_ground_truth)
ground_truth_df.rename(columns={'windspeed_10m': 'gt_windspeed', 'winddirection_10m': 'gt_winddirection'}, inplace=True)

# --- 2. Fetch Historical Forecast Data for Different Models ---
print("Fetching forecast data...")
forecast_api_url = "https://api.open-meteo.com/v1/forecast"
forecast_models = ["arome_france", "gfs_seamless"]
forecast_dfs = {}

for model in forecast_models:
    print(f"  - Fetching {model}...")
    params_forecast = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "windspeed_10m,winddirection_10m",
        "models": model
    }
    df = fetch_weather_data(forecast_api_url, params_forecast)
    if df is not None:
        forecast_dfs[model] = df.rename(columns={'windspeed_10m': f'{model}_windspeed', 'winddirection_10m': f'{model}_winddirection'})

# --- 3. Combine all data into a single DataFrame ---
print("Combining all data sources...")
if ground_truth_df is not None and forecast_dfs:
    final_df = ground_truth_df.copy()
    for model_name, forecast_df in forecast_dfs.items():
        final_df = final_df.join(forecast_df, how='inner')

    # Drop rows with any missing values
    final_df.dropna(inplace=True)
    print("Data successfully downloaded and combined.")
    print(final_df.head())
else:
    print("Failed to download or combine data. Exiting.")
    exit()

# --- Exploratory Data Analysis ---

print("\n--- Starting Exploratory Data Analysis ---")
print(final_df.describe())

# 1. Time Series Plot of Wind Speed
plt.figure(figsize=(15, 8))
plt.plot(final_df.index, final_df['gt_windspeed'], label='Ground Truth', alpha=0.8)
plt.plot(final_df.index, final_df['arome_france_windspeed'], label='AROME Forecast', alpha=0.7)
plt.plot(final_df.index, final_df['gfs_seamless_windspeed'], label='GFS Forecast', alpha=0.7)
plt.title('Wind Speed: Ground Truth vs. Forecast Models (Sample)')
plt.xlabel('Date')
plt.ylabel('Wind Speed (km/h)')
plt.legend()
# Limit to one month for better readability
plt.xlim(final_df.index[0], final_df.index[0] + pd.Timedelta(days=12))
plt.show()

############################
# 2. Distribution of Wind Speed (Histograms)
############################

plt.figure(figsize=(12, 6))
sns.histplot(final_df['gt_windspeed'], color='black', label='Ground Truth', kde=True, stat="density")
sns.histplot(final_df['arome_france_windspeed'], color='blue', label='AROME', kde=True, stat="density")
sns.histplot(final_df['gfs_seamless_windspeed'], color='green', label='GFS', kde=True, stat="density")
plt.title('Distribution of Wind Speeds')
plt.xlabel('Wind Speed (km/h)')
plt.legend()
plt.show()

# 3. Handling Wind Direction (Circular Data)
# For analysis, we convert angles to vector components (U, V)
def to_vector(df):
    for col in df.columns:
        if 'winddirection' in col:
            rad = np.deg2rad(df[col])
            df[col.replace('direction', '_u')] = -np.sin(rad)
            df[col.replace('direction', '_v')] = -np.cos(rad)
    return df

df_vectors = final_df.copy()
df_vectors = to_vector(df_vectors)

# 4. Scatter Plots to show correlation between forecasts and ground truth
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(ax=axes[0], x='gt_windspeed', y='arome_france_windspeed', data=df_vectors, alpha=0.5)
axes[0].plot([0, 80], [0, 80], 'r--') # Perfect forecast line
axes[0].set_title('AROME vs. Ground Truth Wind Speed')
axes[0].set_xlabel('Ground Truth Speed (km/h)')
axes[0].set_ylabel('AROME Forecasted Speed (km/h)')

sns.scatterplot(ax=axes[1], x='gt_windspeed', y='gfs_seamless_windspeed', data=df_vectors, alpha=0.5)
axes[1].plot([0, 80], [0, 80], 'r--')
axes[1].set_title('GFS vs. Ground Truth Wind Speed')
axes[1].set_xlabel('Ground Truth Speed (km/h)')
axes[1].set_ylabel('GFS Forecasted Speed (km/h)')
plt.tight_layout()
plt.show()


# 5. Wind Rose for Direction
# A proper wind rose requires a specialized library, but a polar plot gives a good idea.
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.hist(np.deg2rad(final_df['gt_winddirection']), bins=36, alpha=0.7, label='Ground Truth')
ax.hist(np.deg2rad(final_df['arome_france_winddirection']), bins=36, alpha=0.6, label='AROME')
plt.title('Wind Direction Distribution (Ground Truth vs. AROME)')
plt.legend()
plt.show()

# --- Exploratory Data Analysis ---

print("\n--- Starting Exploratory Data Analysis ---")
print(final_df.describe())

# 1. Time Series Plot of Wind Speed
plt.figure(figsize=(15, 8))
plt.plot(final_df.index, final_df['gt_windspeed'], label='Ground Truth', alpha=0.8)
plt.plot(final_df.index, final_df['arome_france_windspeed'], label='AROME Forecast', alpha=0.7)
plt.plot(final_df.index, final_df['gfs_seamless_windspeed'], label='GFS Forecast', alpha=0.7)
plt.title('Wind Speed: Ground Truth vs. Forecast Models (Sample)')
plt.xlabel('Date')
plt.ylabel('Wind Speed (km/h)')
plt.legend()
# Limit to one month for better readability
plt.xlim(final_df.index[0], final_df.index[0] + pd.Timedelta(days=12))
plt.show()


# 2. Distribution of Wind Speed (Histograms)
plt.figure(figsize=(12, 6))
sns.histplot(final_df['gt_windspeed'], color='black', label='Ground Truth', kde=True, stat="density")
sns.histplot(final_df['arome_france_windspeed'], color='blue', label='AROME', kde=True, stat="density")
sns.histplot(final_df['gfs_seamless_windspeed'], color='green', label='GFS', kde=True, stat="density")
plt.title('Distribution of Wind Speeds')
plt.xlabel('Wind Speed (km/h)')
plt.legend()
plt.show()

# 3. Handling Wind Direction (Circular Data)
# For analysis, we convert angles to vector components (U, V)
def to_vector(df):
    for col in df.columns:
        if 'winddirection' in col:
            rad = np.deg2rad(df[col])
            df[col.replace('direction', '_u')] = -np.sin(rad)
            df[col.replace('direction', '_v')] = -np.cos(rad)
    return df

df_vectors = final_df.copy()
df_vectors = to_vector(df_vectors)

# 4. Scatter Plots to show correlation between forecasts and ground truth
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(ax=axes[0], x='gt_windspeed', y='arome_france_windspeed', data=df_vectors, alpha=0.5)
axes[0].plot([0, 80], [0, 80], 'r--') # Perfect forecast line
axes[0].set_title('AROME vs. Ground Truth Wind Speed')
axes[0].set_xlabel('Ground Truth Speed (km/h)')
axes[0].set_ylabel('AROME Forecasted Speed (km/h)')

sns.scatterplot(ax=axes[1], x='gt_windspeed', y='gfs_seamless_windspeed', data=df_vectors, alpha=0.5)
axes[1].plot([0, 80], [0, 80], 'r--')
axes[1].set_title('GFS vs. Ground Truth Wind Speed')
axes[1].set_xlabel('Ground Truth Speed (km/h)')
axes[1].set_ylabel('GFS Forecasted Speed (km/h)')
plt.tight_layout()
plt.show()


# 5. Wind Rose for Direction
# A proper wind rose requires a specialized library, but a polar plot gives a good idea.
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.hist(np.deg2rad(final_df['gt_winddirection']), bins=36, alpha=0.7, label='Ground Truth')
ax.hist(np.deg2rad(final_df['arome_france_winddirection']), bins=36, alpha=0.6, label='AROME')
plt.title('Wind Direction Distribution (Ground Truth vs. AROME)')
plt.legend()
plt.show()

############################
# 3. Model Evaluation
############################

# --- Scikit-learn: Performance Evaluation ---
print("\n--- Evaluating Models with Scikit-learn ---")

def to_vector(df):
    """
    Calculates u and v wind components from speed and direction columns.
    It correctly infers column names and uses speed in the calculation.
    """
    # Find all wind direction columns
    direction_cols = [col for col in df.columns if 'winddirection' in col]
    
    for dir_col in direction_cols:
        # Infer the corresponding speed column and the prefix
        speed_col = dir_col.replace('winddirection', 'windspeed')
        prefix = dir_col.replace('winddirection', '')
        
        # Check if the matching speed column exists
        if speed_col in df.columns:
            # Get the actual speed and direction data
            speed = df[speed_col]
            direction_rad = np.deg2rad(df[dir_col])
            
            # Calculate u and v components (the correct way)
            # The negative signs are a meteorological convention for "wind from"
            df[f'{prefix}u'] = -speed * np.sin(direction_rad)
            df[f'{prefix}v'] = -speed * np.cos(direction_rad)
        else:
            print(f"Warning: Found direction column '{dir_col}' but no matching speed column '{speed_col}'. Skipping.")
    return df

# Create the new DataFrame with the correct vectors
df_vectors = final_df.copy()
df_vectors = to_vector(df_vectors)

# Now check if the columns were created successfully
print("\nColumns created after vector conversion:")
print([col for col in df_vectors.columns if '_u' in col or '_v' in col])


# 4. Scatter Plots to show correlation between forecasts and ground truth
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(ax=axes[0], x='gt_windspeed', y='arome_france_windspeed', data=df_vectors, alpha=0.5)
axes[0].plot([0, 80], [0, 80], 'r--') # Perfect forecast line
axes[0].set_title('AROME vs. Ground Truth Wind Speed')
axes[0].set_xlabel('Ground Truth Speed (km/h)')
axes[0].set_ylabel('AROME Forecasted Speed (km/h)')

sns.scatterplot(ax=axes[1], x='gt_windspeed', y='gfs_seamless_windspeed', data=df_vectors, alpha=0.5)
axes[1].plot([0, 80], [0, 80], 'r--')
axes[1].set_title('GFS vs. Ground Truth Wind Speed')
axes[1].set_xlabel('Ground Truth Speed (km/h)')
axes[1].set_ylabel('GFS Forecasted Speed (km/h)')
plt.tight_layout()
plt.show()


# We use the vector components for direction error calculation
y_true_speed = df_vectors['gt_windspeed']
y_true_u = df_vectors['gt_u']
y_true_v = df_vectors['gt_v']

results = {}

for model in forecast_models:
    # Speed evaluation
    y_pred_speed = df_vectors[f'{model}_windspeed']
    mae_speed = mean_absolute_error(y_true_speed, y_pred_speed)
    rmse_speed = np.sqrt(mean_squared_error(y_true_speed, y_pred_speed))
    
    # Direction evaluation (using vectors)
    y_pred_u = df_vectors[f'{model}_u']
    y_pred_v = df_vectors[f'{model}_v']
    
    # Vectorial difference error
    # This measures the magnitude of the error vector between true and predicted wind
    direction_error = np.sqrt((y_true_u - y_pred_u)**2 + (y_true_v - y_pred_v)**2)
    mae_direction = direction_error.mean()
    
    results[model] = {'MAE_Speed': mae_speed, 'RMSE_Speed': rmse_speed, 'MAE_Direction_Vector': mae_direction}

# Display results
results_df = pd.DataFrame(results).T
print("\nEvaluation Metrics (Lower is better):")
print(results_df.sort_values(by='RMSE_Speed'))

############################
# 4. Meta-Forecast Evaluation 
############################

# --- PyTorch: Building a Corrective Model ---
print("\n--- Building a Corrective Model with PyTorch ---")

# 1. Prepare Data
# Features are the predictions from all models
features = df_vectors[[f'{model}_windspeed' for model in forecast_models] +
                      [f'{model}_u' for model in forecast_models] +
                      [f'{model}_v' for model in forecast_models]].values

# Targets are the ground truth values
targets = df_vectors[['gt_windspeed', 'gt_u', 'gt_v']].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
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
input_size = X_train.shape[1]
output_size = y_train.shape[1]
model = WindNet(input_size, output_size)

criterion = nn.MSELoss() # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 4. Evaluate the PyTorch Model
with torch.no_grad():
    model.eval()
    y_pred_tensor = model(X_test_tensor)
    y_pred_pytorch = y_pred_tensor.numpy()

# Calculate metrics for PyTorch model
mae_speed_pytorch = mean_absolute_error(y_test[:, 0], y_pred_pytorch[:, 0])
rmse_speed_pytorch = np.sqrt(mean_squared_error(y_test[:, 0], y_pred_pytorch[:, 0]))

direction_error_pytorch = np.sqrt((y_test[:, 1] - y_pred_pytorch[:, 1])**2 + (y_test[:, 2] - y_pred_pytorch[:, 2])**2)
mae_direction_pytorch = direction_error_pytorch.mean()

results['PyTorch_Ensemble'] = {
    'MAE_Speed': mae_speed_pytorch,
    'RMSE_Speed': rmse_speed_pytorch,
    'MAE_Direction_Vector': mae_direction_pytorch
}

# Display final comparison
final_results_df = pd.DataFrame(results).T
print("\n--- Final Comparison Including PyTorch Model ---")
print(final_results_df.sort_values(by='RMSE_Speed'))

############################
# 4. Predict Day after
############################

