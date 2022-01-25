import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from IPython import get_ipython

class model_convolutional_normalized_in_out:
    def preprocess_in(data_in):
        preprocessed_in = data_container.normalize_input(data_in)
        preprocessed_in = data_container.two_dim_input(preprocessed_in)
        preprocessed_in = tensorflow.expand_dims(preprocessed_in, 3)
        
        return preprocessed_in

    def learn(self, data_in, data_out, kernel_size=12, filters=14, epochs=10, dense_layer_size=128):
        preprocessed_in = model_convolutional_normalized_in_out.preprocess_in(data_in)
        preprocessed_out = data_container.normalize_output(data_out)
        preprocessed_out = data_container.one_hot_encode(preprocessed_out)

        # self.model = keras.Sequential()
        # self.model.add(layers.Conv2D(kernel_size=kernel_size, filters=filters, activation="relu", input_shape=(28, 28, 1), padding="same"))
        # self.model.add(layers.Flatten())
        # self.model.add(layers.Dense(10, activation="softmax"))
        
        layer = layers.Conv2D(kernel_size=8, filters=8, activation="relu", input_shape=(28, 28, 1))
        get_ipython().push({'layer': layer})
        
        self.model = keras.Sequential()
        self.model.add(layer)
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(10, activation="softmax"))
        
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        
        self.model.fit(x=preprocessed_in, y=preprocessed_out, epochs=epochs)

    def predict(self, data_in):
        preprocessed_in = model_convolutional_normalized_in_out.preprocess_in(data_in)
        get_ipython().push({'preprocessed_in': preprocessed_in})

        predictions = self.model.predict(x=preprocessed_in)
        predictions = data_container.inverse_normalize_output(predictions)
        predictions = data_container.inverse_one_hot_encode(predictions)
        
        return predictions
    
    # https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a