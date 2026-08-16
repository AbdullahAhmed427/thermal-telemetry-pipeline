import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# data normalization
print("Loading dataset...")
df = pd.read_csv("rtx4050_telemetry_master.csv")

# drop the timestamp string (Neural networks only read numbers)
df = df.drop(columns=['timestamp'])

# normalize all features to a 0-1 scale
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# time-series feature enginerring
print("Engineering sliding windows...")
LOOKBACK = 60  # context window: past 60 seconds
FORECAST = 30  # prediction horizon: 30 seconds into the future

X, y = [], []
# get the column index for the target (gpu_temp_c)
temp_idx = df.columns.get_loc('gpu_temp_c')

# slide the window across the entire 26-minute dataset
for i in range(LOOKBACK, len(scaled_data) - FORECAST):
    # X: 60 rows of all columns
    X.append(scaled_data[i - LOOKBACK:i, :])
    # y: 1 value (the temperature 30 seconds ahead)
    y.append(scaled_data[i + FORECAST, temp_idx])

X = np.array(X)
y = np.array(y)

print(f"\nFeature Tensor (X) Shape: {X.shape}") 


print(f"Target Tensor (y) Shape: {y.shape}")


# drafting LSTM architecture
print("\nCompiling LSTM Model Architecture...")

model = Sequential([
    # Layer 1: Ingests the 60-second window. 
    # return_sequences=True passes the entire time context to the next layer.
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2), # randomly drops 20% of connections to prevent overfitting
    
    # Layer 2: Compresses the time context into core thermal patterns
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    
    # Layer 3: Interpretation layer
    Dense(16, activation='relu'),
    
    # Output Layer: 1 continuous numerical value (The Predicted Scaled Temperature)
    Dense(1, activation='linear')
])

# Adam optimizer and Mean Squared Error for regression tasks
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# print the blueprint of the neural network
model.summary()





print("\nSplitting Data (80% Train, 20% Validation)")
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print("Training the LSTM Model")

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

print("\nTraining Complete! Saving model to disk...")
model.save('thermal_lstm_model.h5')


plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss (MSE)', color='tab:blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='tab:orange', linewidth=2)
plt.title('LSTM Training Performance: Thermal Prediction', fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (Scaled)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()