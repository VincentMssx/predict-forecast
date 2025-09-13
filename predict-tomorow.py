import pandas as pd
import requests
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

# --- 1. Define the Model Architecture (must match the trained model) ---
# We need to define the class again so PyTorch knows how to load the weights.
class MetaForecastGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob):
        super(MetaForecastGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out)

# --- 2. Configuration and Helper Functions ---
TARGET_DATE = "2025-09-12"
YESTERDAY = pd.to_datetime(TARGET_DATE) - pd.Timedelta(days=1)
YESTERDAY_STR = YESTERDAY.strftime('%Y-%m-%d')

LATITUDE = 46.244
LONGITUDE = -1.561
MODEL_PATH = "meta_forecast_model.pth"
SCALER_PATH = "data_scaler.joblib"

def fetch_weather_data(api_url, params):
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def to_vector(df, prefix):
    rad = np.deg2rad(df[f'{prefix}_winddirection'])
    speed = df[f'{prefix}_windspeed']
    df[f'{prefix}_u'] = -speed * np.sin(rad)
    df[f'{prefix}_v'] = -speed * np.cos(rad)
    return df

def to_speed_dir(df, prefix):
    u = df[f'{prefix}_u']
    v = df[f'{prefix}_v']
    df[f'{prefix}_windspeed'] = np.sqrt(u**2 + v**2)
    # arctan2 is crucial for getting the correct angle in all quadrants
    rad = np.arctan2(-u, -v)
    deg = np.rad2deg(rad)
    df[f'{prefix}_winddirection'] = (deg + 360) % 360 # Ensure positive degrees
    return df

# --- 3. Load Model and Scaler ---
print("Loading saved model and scaler...")
scaler = joblib.load(SCALER_PATH)

# Instantiate the model with the same parameters as during training
model = MetaForecastGRU(input_size=6, hidden_size=1024, output_size=2, num_layers=2, dropout_prob=0.2)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # Set model to evaluation mode (very important!)
print("Model and scaler loaded successfully.")

# --- 4. Fetch Necessary Data for Prediction ---
print(f"Fetching data for target date {TARGET_DATE} and previous day {YESTERDAY_STR}...")

# A. Get Ground Truth for YESTERDAY (from the archive)
archive_api = "https://archive-api.open-meteo.com/v1/archive"
params_gt = {"latitude": LATITUDE, "longitude": LONGITUDE, "start_date": YESTERDAY_STR, "end_date": YESTERDAY_STR, "hourly": "windspeed_10m,winddirection_10m"}
df_gt_yesterday = fetch_weather_data(archive_api, params_gt)
df_gt_yesterday = to_vector(df_gt_yesterday.rename(columns={'windspeed_10m': 'gt_windspeed', 'winddirection_10m': 'gt_winddirection'}), 'gt')

# B. Get Forecasts for TOMORROW (from the live forecast API)
forecast_api = "https://api.open-meteo.com/v1/forecast"
params_fc = {"latitude": LATITUDE, "longitude": LONGITUDE, "start_date": TARGET_DATE, "end_date": TARGET_DATE, "hourly": "windspeed_10m,winddirection_10m"}
df_arome = fetch_weather_data(forecast_api, {**params_fc, "models": "arome_france"})
df_arome = to_vector(df_arome.rename(columns={'windspeed_10m': 'arome_windspeed', 'winddirection_10m': 'arome_winddirection'}), 'arome')
df_gfs = fetch_weather_data(forecast_api, {**params_fc, "models": "gfs_seamless"})
df_gfs = to_vector(df_gfs.rename(columns={'windspeed_10m': 'gfs_windspeed', 'winddirection_10m': 'gfs_winddirection'}), 'gfs')

# --- 5. Prepare the Input Tensor for the Model ---
print("Preparing input tensor...")

# Extract the NumPy arrays for each component
past_gt_vals = df_gt_yesterday[['gt_u', 'gt_v']].values
future_arome_vals = df_arome[['arome_u', 'arome_v']].values
future_gfs_vals = df_gfs[['gfs_u', 'gfs_v']].values

# Concatenate along the columns (axis=1) to create the (24, 6) feature matrix
# This correctly mimics the logic from the training script
input_data = np.concatenate([past_gt_vals, future_arome_vals, future_gfs_vals], axis=1)

# Scale the data using the LOADED scaler
# The scaler expects a 2D array, and our `input_data` is already in the correct shape
scaled_input_data = scaler.transform(input_data)

# Reshape to the (batch_size, sequence_length, features) format for the GRU
input_tensor = torch.tensor(scaled_input_data, dtype=torch.float32).unsqueeze(0) # unsqueeze(0) adds the batch dimension
print(f"Input tensor shape: {input_tensor.shape}") # Should be torch.Size([1, 24, 6])

# --- 6. Make the Prediction ---
print("Generating meta-forecast...")
with torch.no_grad():
    scaled_prediction = model(input_tensor) # Shape will be (1, 24, 2)

# --- 7. Post-process the Output (This part should now work) ---
# Squeeze to remove batch dim, then convert to numpy. Shape becomes (24, 2)
prediction_numpy = scaled_prediction.squeeze(0).numpy()

# Inverse transform the scaled prediction to get real u/v values
dummy_array = np.zeros((24, 6)) # Dummy array matching the scaler's 6 features
dummy_array[:, :2] = prediction_numpy # Place our prediction in the first two columns
unscaled_prediction = scaler.inverse_transform(dummy_array)[:, :2] # Inverse transform and take only our 2 columns

# Create a final DataFrame with the results
prediction_df = pd.DataFrame(unscaled_prediction, columns=['meta_u', 'meta_v'], index=df_arome.index)
prediction_df = to_speed_dir(prediction_df, 'meta')

# Add raw forecasts for comparison
prediction_df['arome_windspeed'] = df_arome['arome_windspeed']
prediction_df['gfs_windspeed'] = df_gfs['gfs_windspeed']

print(f"\n--- Meta-Forecast for {TARGET_DATE} ---")
print(prediction_df[['meta_windspeed', 'meta_winddirection']].round(2))

# --- 8. Visualize the Prediction ---
plt.figure(figsize=(16, 9))
plt.plot(prediction_df.index.hour, prediction_df['meta_windspeed'], label='Meta-Forecast (Our Model)', color='green', linewidth=2.5, marker='o')
plt.plot(prediction_df.index.hour, prediction_df['arome_windspeed'], label='AROME Raw Forecast', color='red', linestyle=':', alpha=0.8)
plt.plot(prediction_df.index.hour, prediction_df['gfs_windspeed'], label='GFS Raw Forecast', color='blue', linestyle=':', alpha=0.8)

plt.title(f'Meta Wind Speed Forecast for {TARGET_DATE}')
plt.xlabel('Hour of the Day')
plt.ylabel('Wind Speed (km/h)')
plt.xticks(np.arange(0, 24, 1))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()