import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Ensure a directory exists for saving models
os.makedirs("saved_models", exist_ok=True)

################################################################################
# 1. Get and Prepare Data (Upgraded with More Features)
################################################################################

# --- Configuration ---
LATITUDE = 46.244
LONGITUDE = -1.561
# Using a longer time period for more robust training
START_DATE = "2020-01-01"
END_DATE = "2025-08-31" # Use a completed period for training

# --- Fetching Functions ---
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

# --- Data Acquisition with Additional Variables ---
print("Fetching historical data with additional weather variables...")
archive_api_url = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "windspeed_10m,winddirection_10m,temperature_2m,relativehumidity_2m,surface_pressure"
base_params = {"latitude": LATITUDE, "longitude": LONGITUDE, "start_date": START_DATE, "end_date": END_DATE, "hourly": HOURLY_VARS}

df_gt = fetch_weather_data(archive_api_url, base_params)
df_arome = fetch_weather_data(archive_api_url, {**base_params, "models": "arome_france"})
df_gfs = fetch_weather_data(archive_api_url, {**base_params, "models": "gfs_seamless"})

# --- Combine, Rename, and Convert to Vectors ---
print("Combining and processing data...")
# Add prefixes to all columns to avoid name clashes
df_gt = df_gt.add_prefix('gt_')
df_arome = df_arome.add_prefix('arome_')
df_gfs = df_gfs.add_prefix('gfs_')

final_df = df_gt.join([df_arome, df_gfs], how='inner').dropna()

def to_vector(df, prefix):
    # This function now only adds u/v, does not modify original df
    rad = np.deg2rad(df[f'{prefix}_winddirection_10m'])
    speed = df[f'{prefix}_windspeed_10m']
    return -speed * np.sin(rad), -speed * np.cos(rad)

final_df['gt_u'], final_df['gt_v'] = to_vector(final_df, 'gt')
final_df['arome_u'], final_df['arome_v'] = to_vector(final_df, 'arome')
final_df['gfs_u'], final_df['gfs_v'] = to_vector(final_df, 'gfs')

# --- Feature Engineering (NEW & IMPROVED) ---
print("Engineering new features...")
# 1. Cyclical Time Features
final_df['hour_sin'] = np.sin(2 * np.pi * final_df.index.hour / 24.0)
final_df['hour_cos'] = np.cos(2 * np.pi * final_df.index.hour / 24.0)
final_df['dayofyear_sin'] = np.sin(2 * np.pi * final_df.index.dayofyear / 365.0)
final_df['dayofyear_cos'] = np.cos(2 * np.pi * final_df.index.dayofyear / 365.0)

# 2. Forecast Interaction Features
final_df['diff_u'] = final_df['arome_u'] - final_df['gfs_u']
final_df['diff_v'] = final_df['arome_v'] - final_df['gfs_v']
final_df['diff_temp'] = final_df['arome_temperature_2m'] - final_df['gfs_temperature_2m']

# --- Feature Selection and Scaling ---
# Define ALL columns that will be used as input features
feature_cols = [
    'gt_u', 'gt_v', 'gt_temperature_2m', 'gt_relativehumidity_2m', 'gt_surface_pressure', # Past GT
    'arome_u', 'arome_v', 'arome_temperature_2m', 'arome_relativehumidity_2m', 'arome_surface_pressure', # AROME Forecast
    'gfs_u', 'gfs_v', 'gfs_temperature_2m', 'gfs_relativehumidity_2m', 'gfs_surface_pressure', # GFS Forecast
    'hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos', # Time Features
    'diff_u', 'diff_v', 'diff_temp' # Interaction Features
]
features_df = final_df[feature_cols]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)
scaled_df = pd.DataFrame(features_scaled, index=features_df.index, columns=features_df.columns)

################################################################################
# 2. Create Time-Series Windows (Upgraded with Residual Target)
################################################################################
def create_dataset_windows(scaled_df, original_df):
    X, y = [], []
    input_cols = scaled_df.columns
    
    for i in tqdm(range(len(scaled_df) - 48), desc="Creating windows"):
        if i % 24 != 0: continue

        # Input is a 48-hour window of ALL features (past GT and future forecasts)
        # Note: We structure it so the model can learn from past GT AND future forecasts
        past_features = scaled_df[input_cols].iloc[i : i+24].values
        future_features = scaled_df[input_cols].iloc[i+24 : i+48].values
        
        # For simplicity, we'll combine them for now. A more advanced model could use an Encoder-Decoder.
        input_window = np.concatenate([past_features, future_features], axis=0)
        
        # TARGET: The residual (error) of the AROME forecast for the next 24 hours
        arome_forecast = original_df[['arome_u', 'arome_v']].iloc[i+24 : i+48].values
        ground_truth = original_df[['gt_u', 'gt_v']].iloc[i+24 : i+48].values
        target_residual = ground_truth - arome_forecast
        
        X.append(input_window)
        y.append(target_residual)
        
    return np.array(X), np.array(y)

print("\nStructuring data into daily windows...")
# We need the original (unscaled) df to calculate the true residual
X, y = create_dataset_windows(scaled_df, final_df)

# We need to scale the target (residuals) as well
# Fit a separate scaler for the target variable
target_scaler = StandardScaler()
y_flat = y.reshape(-1, y.shape[-1])
y_scaled_flat = target_scaler.fit_transform(y_flat)
y_scaled = y_scaled_flat.reshape(y.shape)

print(f"Created {X.shape[0]} daily samples.")
print(f"Shape of input sample (X): {X.shape[1]} hours x {X.shape[2]} features")
print(f"Shape of target sample (y): {y_scaled.shape[1]} hours x {y_scaled.shape[2]} values")

# --- Split into Train, Validation, and Test sets ---
X_train, X_temp, y_train, y_temp = train_test_split(X, y_scaled, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# --- Create DataLoaders ---
def to_tensor(data): return torch.tensor(data, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(to_tensor(X_train), to_tensor(y_train)), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(to_tensor(X_val), to_tensor(y_val)), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(to_tensor(X_test), to_tensor(y_test)), batch_size=32, shuffle=False)

################################################################################
# 3. Define and Instantiate the GRU Model
################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nRunning on device: {device}")

class MetaForecastGRU(nn.Module):
    # This is a simplified Seq2Seq model where output length = input length
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob):
        super(MetaForecastGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        gru_out, _ = self.gru(x)
        # We only need to predict for the last 24 hours of the sequence
        output = self.fc(gru_out[:, -24:, :]) 
        return output

# --- Hyperparameters ---
INPUT_FEATURES = X.shape[2]
HIDDEN_SIZE = 512 # Reduced size to prevent overfitting with more features
OUTPUT_SIZE = y.shape[2]
NUM_LAYERS = 2
DROPOUT = 0.3 # Increased dropout for regularization
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

model = MetaForecastGRU(INPUT_FEATURES, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DROPOUT).to(device)
print("\nModel Architecture:")
print(model)

################################################################################
# 4. Train the Model with Scheduler and Early Stopping (NEW & IMPROVED)
################################################################################
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

# Early stopping parameters
patience = 7
best_val_loss = float('inf')
epochs_no_improve = 0
BEST_MODEL_PATH = "saved_models/best_meta_forecast_model.pth"

print(f"\nStarting training for up to {NUM_EPOCHS} epochs with early stopping...")
for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
    
    # Update learning rate scheduler
    scheduler.step(avg_val_loss)
    
    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Validation loss improved. Saving best model to {BEST_MODEL_PATH}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
        break

################################################################################
# 5. Evaluate the BEST Model on the Test Set
################################################################################
print(f"\nLoading best model from {BEST_MODEL_PATH} for final evaluation...")
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

all_preds_scaled = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_preds_scaled.append(outputs.cpu().numpy())

# Inverse transform the scaled predictions (residuals)
y_pred_scaled_flat = np.concatenate(all_preds_scaled).reshape(-1, OUTPUT_SIZE)
y_pred_residuals = target_scaler.inverse_transform(y_pred_scaled_flat)

# --- Reconstruct the final meta-forecast ---
# Find the original AROME forecasts that correspond to the test set
test_indices = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)[1]
test_indices_final = train_test_split(test_indices, test_size=0.5, random_state=42)[1]

arome_forecasts_test = []
gt_test = []
for idx in test_indices_final:
    start = idx * 24 + 24
    end = start + 24
    arome_forecasts_test.append(final_df[['arome_u', 'arome_v']].iloc[start:end].values)
    gt_test.append(final_df[['gt_u', 'gt_v']].iloc[start:end].values)

arome_forecasts_test = np.concatenate(arome_forecasts_test)
gt_test_flat = np.concatenate(gt_test)

# Final Meta-Forecast = AROME Forecast + Predicted Residual
meta_forecast_final = arome_forecasts_test + y_pred_residuals

# --- Calculate Final Metrics ---
mae = mean_absolute_error(gt_test_flat, meta_forecast_final)
rmse = np.sqrt(mean_squared_error(gt_test_flat, meta_forecast_final))
print(f"\nFinal Meta-Forecast MAE on u/v components: {mae:.4f}")
print(f"Final Meta-Forecast RMSE on u/v components: {rmse:.4f}")

################################################################################
# 6. SAVE THE FINAL SCALERS
################################################################################
print("\nSaving scalers to disk...")
joblib.dump(scaler, "saved_models/feature_scaler.joblib")
joblib.dump(target_scaler, "saved_models/target_scaler.joblib")
print("Artifacts are ready for prediction.")