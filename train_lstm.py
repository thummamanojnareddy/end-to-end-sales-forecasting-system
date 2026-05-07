import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length=10):

    X = []
    y = []

    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

    return np.array(X), np.array(y)

def train_lstm(train):

    scaler = MinMaxScaler()

    scaled_train = scaler.fit_transform(
        train[['sales']]
    )

    X_train, y_train = create_sequences(scaled_train)

    model = Sequential()

    model.add(
        LSTM(
            64,
            activation='relu',
            input_shape=(X_train.shape[1], 1)
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=16,
        verbose=1
    )

    model.save('models/lstm.h5')

    return model
