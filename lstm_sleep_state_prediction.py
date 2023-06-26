import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
data_path = "01202023-Heart Rate and Sleep Stage - For RNN.csv"
data = pd.read_csv(data_path, parse_dates=["date"], index_col="date")
scaler = MinMaxScaler(feature_range=(0, 1))
data['heart_rate'] = scaler.fit_transform(data['heart_rate'].values.reshape(-1, 1))
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Prepare data for the LSTM model
def create_lstm_data(dataset, window_size):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset['heart_rate'].iloc[i:i+window_size].values)
        y.append(dataset['sleep_state'].iloc[i+window_size])
    return np.array(X), np.array(y)

window_size = 10
X_train, y_train = create_lstm_data(train_data, window_size)
X_test, y_test = create_lstm_data(test_data, window_size)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_train = to_categorical(y_train, num_classes=6)
y_test = to_categorical(y_test, num_classes=6)

# Build and train the LSTM model
model = Sequential([
    LSTM(50, activation="relu", input_shape=(window_size, 1)),
    Dense(6, activation="softmax")
])

model.compile(optimizer=Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")

# Evaluate the model
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")

# Save the model
model.save('my_lstm_model.h5')

# Make predictions
y_train_pred = np.argmax(model.predict(X_train), axis=-1)
y_test_pred = np.argmax(model.predict(X_test), axis=-1)

# Classification report
print("\nTrain Classification Report:")
print(classification_report(np.argmax(y_train, axis=1), y_train_pred))
print("\nTest Classification Report:")
print(classification_report(np.argmax(y_test, axis=1), y_test_pred))

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

plot_confusion_matrix(np.argmax(y_train, axis=1), y_train_pred, title="Train Confusion Matrix")
plot_confusion_matrix(np.argmax(y_test, axis=1), y_test_pred, title="Test Confusion Matrix")



