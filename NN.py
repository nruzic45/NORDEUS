from keras import Sequential
from keras.layers import Dense

def baseline_model():
    # create model
    model = Sequential()

    model.add(Dense(16, input_dim=7, activation='relu'))

    model.add(Dense(8, input_dim=7, activation='relu'))

    model.add(Dense(1))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
