import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('my_lstm_model.h5')

# Load and preprocess the new data
new_data_path = "path_to_new_data.csv"
new_data = pd.read_csv(new_data_path, parse_dates=["date"], index_col="date")
scaler = MinMaxScaler(feature_range=(0, 1))
new_data['heart_rate'] = scaler.fit_transform(new_data['heart_rate'].values.reshape(-1, 1))

# Prepare data for the LSTM model (using the same function as in the training script)
def create_lstm_data(dataset, window_size):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset['heart_rate'].iloc[i:i+window_size].values)
        y.append(dataset['sleep_state'].iloc[i+window_size])
    return np.array(X), np.array(y)

window_size = 10
X_new, y_new = create_lstm_data(new_data, window_size)
X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))

# Make predictions
y_new_pred = np.argmax(model.predict(X_new), axis=-1)

# Print the predicted sleep states
print("Predicted Sleep States:", y_new_pred)
