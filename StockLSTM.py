import helper
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

# Load the dataset
df = pd.read_csv('TSLA.csv')

# Extract the 'Open' values and reshape
df = df['Open'].values
df = df.reshape(-1, 1)

# Setup datasets
dataset_train = np.array(df[:int(df.shape[0] * 0.8)])
dataset_test = np.array(df[int(df.shape[0] * 0.8):])

# Scale the values
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)

# Define create_dataset function
def create_dataset(data, time_step=50):
    x, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        x.append(a)
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)

# Use the 'create_dataset' function to create train/test datasets
x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)

# Ensure x_train and x_test are numpy arrays before reshaping
x_train = np.array(x_train)
x_test = np.array(x_test)

# Check the shape of x_train before reshaping
print("Shape of x_train before reshaping:", np.shape(x_train))  # Expected output should be (352, 50)

# Reshape the data to fit the LSTM layer input: [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Check the shape of x_train after reshaping
print("Shape of x_train after reshaping:", np.shape(x_train))  # Expected output should be (352, 50, 1)

# Model creation
model = Sequential()

# Adding the first LSTM layer with Dropout and return_sequences=True
model.add(LSTM(units=4, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

# Adding the second LSTM layer with Dropout
model.add(LSTM(units=4))
model.add(Dropout(0.2))

# Adding the final Dense layer
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(x_train, y_train, epochs=5, batch_size=16, verbose=0)

# Printing the model summary
model.summary()

# Making predictions
predictions = model.predict(x_test)

# Printing the predictions
print(predictions)
