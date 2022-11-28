import keras
from keras.applications.densenet import layers


def Neural(x_training_data, y_training_data, epoha=50, lr=0.01, akt='relu', opti='Adam', aktIz='relu'):
    input_shape = [x_training_data.shape[1]]

    modelN = keras.Sequential([
        layers.Dense(units=512, activation=akt, input_shape=input_shape),
        layers.Dense(units=256, activation=akt),
        layers.Dense(units=256, activation=akt),
        layers.Dense(13, activation=aktIz),

    ])

    modelN.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="mae",
        metrics=["accuracy"]
    )

    izlaz = modelN.fit(
        x_training_data, y_training_data,
        batch_size=64,
        epochs=epoha,
    )

    return modelN
