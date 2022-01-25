from data_container import data_container
from tensorflow import keras
from tensorflow.keras import layers
from IPython import get_ipython

class model_sequential_normalized_in_out:
    def __init__(self):
        self.model = keras.Sequential(
            [
                layers.InputLayer(28*28),
                layers.Dense(300, activation="relu"),
                layers.Dense(300, activation="relu"),
                layers.Dense(300, activation="relu"),
                layers.Dense(300, activation="relu"),
                layers.Dense(1, activation="relu")
            ]
        )

        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def learn(self, data_in, data_out):
        preprocessed_in = data_container.normalize_input(data_in)
        preprocessed_out = data_container.normalize_output(data_out)

        self.model.fit(x=preprocessed_in, y=preprocessed_out, epochs=100, batch_size=100)

    def predict(self, data_in):
        preprocessed_in = data_container.normalize_input(data_in)
        get_ipython().push({'predict_preprocessed_in_test': preprocessed_in})
        predictions = self.model.predict(x=preprocessed_in)
        return data_container.inverse_normalize_output(predictions)