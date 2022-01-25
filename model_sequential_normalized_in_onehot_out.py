from data_container import data_container
from tensorflow import keras
from tensorflow.keras import layers

class model_sequential_normalized_in_onehot_out:
    def __init__(self):
        self.model = keras.Sequential(
            [
                layers.InputLayer(28*28),
                layers.Dense(28*28, activation="relu"),
                layers.Dense(28*28, activation="relu"),
                layers.Dense(28*28, activation="relu"),
                layers.Dense(28*28, activation="relu"),
                layers.Dense(10, activation="softmax")
            ]
        )

        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def learn(self, data_in, data_out):
        preprocessed_in = data_container.normalize_input(data_in)
        preprocessed_out = data_container.one_hot_encode(data_out)

        self.model.fit(x=preprocessed_in, y=preprocessed_out, epochs=30, batch_size=100)

    def predict(self, data_in):
        predictions = self.model.predict(x=data_in)
        return data_container.inverse_one_hot_encode(predictions)