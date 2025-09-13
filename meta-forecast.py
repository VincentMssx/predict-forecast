import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

################################################################################
# 1. Get and Prepare Data
################################################################################

# --- Configuration ---
LATITUDE = 46.244
LONGITUDE = -1.561
# A "big" network requires more data. Let's fetch 2 years.
START_DATE = "2019-01-01"
END_DATE = "2025-09-13"

# --- Fetching Functions (Unchanged) ---
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

# --- Data Acquisition ---
print("Fetching historical data...")
archive_api_url = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "windspeed_10m,winddirection_10m,temperature_2m,relativehumidity_2m,surface_pressure"
params_gt = {"latitude": LATITUDE, "longitude": LONGITUDE, "start_date": START_DATE, "end_date": END_DATE, "hourly": "windspeed_10m,winddirection_10m", "hourly": HOURLY_VARS}
df_gt = fetch_weather_data(archive_api_url, params_gt)
df_gt.rename(columns={'windspeed_10m': 'gt_windspeed', 'winddirection_10m': 'gt_winddirection'}, inplace=True)

params_arome = {**params_gt, "models": "arome_france"}
df_arome = fetch_weather_data(archive_api_url, params_arome)
df_arome.rename(columns={'windspeed_10m': 'arome_windspeed', 'winddirection_10m': 'arome_winddirection'}, inplace=True)

params_gfs = {**params_gt, "models": "gfs_seamless"}
df_gfs = fetch_weather_data(archive_api_url, params_gfs)
df_gfs.rename(columns={'windspeed_10m': 'gfs_windspeed', 'winddirection_10m': 'gfs_winddirection'}, inplace=True)

# --- Combine and Convert to Vectors ---
print("Combining and processing data...")
final_df = df_gt.join([df_arome, df_gfs], how='inner').dropna()

def to_vector(df, prefix):
    speed_col = f'{prefix}_windspeed'
    dir_col = f'{prefix}_winddirection'
    rad = np.deg2rad(df[dir_col])
    speed = df[speed_col]
    df[f'{prefix}_u'] = -speed * np.sin(rad)
    df[f'{prefix}_v'] = -speed * np.cos(rad)
    return df

final_df = to_vector(final_df, 'gt')
final_df = to_vector(final_df, 'arome')
final_df = to_vector(final_df, 'gfs')

# --- Feature Selection and Scaling ---
features_df = final_df[['gt_u', 'gt_v', 'arome_u', 'arome_v', 'gfs_u', 'gfs_v']]

# Scaling is crucial for neural networks
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)
scaled_df = pd.DataFrame(features_scaled, index=features_df.index, columns=features_df.columns)

################################################################################
# 2. Create Time-Series Windows (Dataset Preparation)
################################################################################
def create_dataset_windows(df):
    X, y = [], []
    # We need 48 hours for each sample (24 past, 24 future)
    # The loop stops at len(df) - 48 to avoid index errors
    for i in tqdm(range(len(df) - 48), desc="Creating windows"):
        # We step by 24 hours to create one sample per day
        if i % 24 != 0:
            continue

        # Input Features: Past 24h of GT + Next 24h of Forecasts
        past_gt = df[['gt_u', 'gt_v']].iloc[i : i+24].values
        future_arome = df[['arome_u', 'arome_v']].iloc[i+24 : i+48].values
        future_gfs = df[['gfs_u', 'gfs_v']].iloc[i+24 : i+48].values
        diff_model = future_arome - future_gfs
        final_df['hour_sin'] = np.sin(2 * np.pi * final_df.index.hour / 24.0)
        final_df['hour_cos'] = np.cos(2 * np.pi * final_df.index.hour / 24.0)
        
        # Combine into a single input sequence for the day
        # Shape: (24, 6) -> 24 hours, 6 features per hour
        input_features = np.concatenate([past_gt, future_arome, future_gfs, diff_model, final_df], axis=1)
        
        # Target: Next 24h of actual Ground Truth
        # Shape: (24, 2) -> 24 hours, 2 target values per hour
        arome_forecast = df[['arome_u', 'arome_v']].iloc[i+24 : i+48].values
        ground_truth = df[['gt_u', 'gt_v']].iloc[i+24 : i+48].values
        target = ground_truth - arome_forecast # This is the "residual"
        
        X.append(input_features)
        y.append(target)
        
    return np.array(X), np.array(y)

print("\nStructuring data into daily windows for the neural network...")
X, y = create_dataset_windows(scaled_df)
print(f"Created {X.shape[0]} daily samples.")
print(f"Shape of a single input sample (X): {X.shape[1]} hours x {X.shape[2]} features")
print(f"Shape of a single target sample (y): {y.shape[1]} hours x {y.shape[2]} values")

# --- Split and Create DataLoaders ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)

################################################################################
# 3. Define the GRU Neural Network
################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.device("cuda:0")
    print("Running on GPU")
else:
    torch.device("cpu")
    print("Running on CPU")
class MetaForecastGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob):
        super(MetaForecastGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # Critical for easy data handling
            dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # gru_out shape: (batch_size, sequence_length, hidden_size)
        gru_out, _ = self.gru(x)
        
        # We apply the linear layer to every timestep of the sequence
        # output shape: (batch_size, sequence_length, output_size)
        output = self.fc(gru_out)
        return output

# --- Model Instantiation ---
INPUT_FEATURES = X.shape[2]    # Should be 6
HIDDEN_SIZE = 1024             # As requested
OUTPUT_SIZE = y.shape[2]       # Should be 2 (u, v)
NUM_LAYERS = 2                 # 2 GRU layers stacked for more depth
DROPOUT = 0.2                  # Regularization to prevent overfitting

model = MetaForecastGRU(INPUT_FEATURES, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DROPOUT)
print("\nModel Architecture:")

################################################################################
# 4. Training the Model
################################################################################
criterion = nn.MSELoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
num_epochs = 50

print(f"\nStarting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

################################################################################
# 5. Evaluation and Visualization
################################################################################
print("\nEvaluating model on the test set...")
model.eval()
all_preds = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        all_preds.append(outputs.numpy())

# --- Inverse Transform and Compare ---
# Reshape predictions and test data back to 2D for inverse scaling
y_pred_flat = np.concatenate(all_preds).reshape(-1, OUTPUT_SIZE)
y_test_flat = y_test.reshape(-1, OUTPUT_SIZE)

# Create dummy arrays to match the scaler's original shape
dummy_pred = np.zeros((len(y_pred_flat), len(features_df.columns)))
dummy_pred[:, :OUTPUT_SIZE] = y_pred_flat
y_pred_unscaled = scaler.inverse_transform(dummy_pred)[:, :OUTPUT_SIZE]

dummy_test = np.zeros((len(y_test_flat), len(features_df.columns)))
dummy_test[:, :OUTPUT_SIZE] = y_test_flat
y_test_unscaled = scaler.inverse_transform(dummy_test)[:, :OUTPUT_SIZE]

# --- Calculate Final Metrics ---
mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
print(f"Meta-Forecast MAE on u/v components: {mae:.4f}")
print(f"Meta-Forecast RMSE on u/v components: {rmse:.4f}")

# --- Visualize a Sample Prediction ---
print("\nVisualizing a 24-hour forecast from the test set...")
# Choose a random day from the test set
sample_idx = np.random.randint(0, len(y_test))
predicted_day = y_pred_unscaled.reshape(len(y_test), 24, 2)[sample_idx]
actual_day = y_test_unscaled.reshape(len(y_test), 24, 2)[sample_idx]

#  Get the original data for this sample day for comparison and date 
start_index_in_df = (sample_idx * 24) + 24 # We skip the first 24h used as past_gt
target_day_df = final_df.iloc[start_index_in_df : start_index_in_df + 24]

# Get the date for the title
target_date = target_day_df.index[0].strftime('%Y-%m-%d')

# Get the raw AROME and GFS forecasts for that same day
arome_day_raw = target_day_df[['arome_u', 'arome_v']].values
gfs_day_raw = target_day_df[['gfs_u', 'gfs_v']].values

# For comparison, get the raw AROME forecast for that same day
# First, find the original index in the main dataframe
original_idx = scaled_df.index.get_loc(scaled_df.iloc[sample_idx*24+24:sample_idx*24+48].index[0])
arome_day_raw = final_df[['arome_u', 'arome_v']].iloc[original_idx:original_idx+24].values

# Convert u,v back to speed for plotting
def to_speed(u, v): return np.sqrt(u**2 + v**2)

predicted_speed = to_speed(predicted_day[:, 0], predicted_day[:, 1])
actual_speed = to_speed(actual_day[:, 0], actual_day[:, 1])
arome_speed_raw = to_speed(arome_day_raw[:, 0], arome_day_raw[:, 1])
gfs_speed_raw = to_speed(gfs_day_raw[:, 0], gfs_day_raw[:, 1])

plt.figure(figsize=(16, 9))
plt.plot(actual_speed, label='Ground Truth (Actual)', color='black', linewidth=2.5, marker='o', markersize=5)
plt.plot(predicted_speed, label='Meta-Forecast (Our Model)', color='green', linestyle='--', linewidth=2, marker='x')
plt.plot(gfs_speed_raw, label='GFS Raw Forecast', color='blue', linestyle=':', alpha=0.8)
plt.plot(arome_speed_raw, label='AROME Raw Forecast', color='red', linestyle=':', alpha=0.8)
plt.title(f'24-Hour Wind Speed Forecast Comparison for {target_date}')
plt.xlabel('Hour of the Day')
plt.ylabel('Wind Speed (km/h)')
plt.xticks(np.arange(0, 24, 1))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

################################################################################
# 6. SAVE THE TRAINED MODEL AND SCALER
################################################################################

print("\nTraining complete. Saving model and scaler to disk...")

# 1. Save the model's state dictionary
MODEL_PATH = "meta_forecast_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# 2. Save the scaler object
SCALER_PATH = "data_scaler.joblib"
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved to {SCALER_PATH}")

print("\nArtifacts are ready for prediction.")


