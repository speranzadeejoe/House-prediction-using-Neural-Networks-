import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Data
df = pd.read_csv(r"/content/kc_house_data.csv")

# Data Preprocessing - Creating house_age
df["reg_year"] = df["date"].str[:4].astype(int)
df["house_age"] = np.where(df['yr_renovated'] == 0,
                           df["reg_year"] - df["yr_built"],
                           df["reg_year"] - df["yr_renovated"])

df.drop(['date', 'yr_built', 'yr_renovated', 'reg_year', 'lat', 'long', 'zipcode'], axis=1, inplace=True)

# Remove invalid ages
df = df[df['house_age'] != -1]

# Set features (x) and target (y)
x = df.drop('price', axis=1)
y = df['price']

# Check columns (for safety)
print("Final columns after preprocessing:")
print(x.columns)

# Scale the features — CRUCIAL for smoother training
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split into train and validation sets (more control than validation_split)
x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.33, random_state=42)

# Build the Model - Simpler, more stable architecture
model = keras.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(x.shape[1],)))  # Input layer for 14 features
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1))  # Output layer (predicting price)

# Compile Model
model.compile(optimizer="adam", loss="mse")

# Train the Model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32)

# Plot Loss Curve (should now be smoother)
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Check number of input features the model expects (safety check)
print(f"Number of features: {x.shape[1]}")

# Define new input data for prediction - after scaling!
Xnew = np.array([[2, 3, 1280, 5550, 1, 0, 0, 4, 7, 2280, 0, 1440, 5750, 60]])

# Scale new data using the same scaler
Xnew_scaled = scaler.transform(Xnew)

# Predict
Ynew = model.predict(Xnew_scaled)

# Print predicted price
print("Predicted house price:", Ynew[0][0])

