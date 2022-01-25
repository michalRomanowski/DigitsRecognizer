import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow
from IPython import get_ipython
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

MAX_IN_VAL = 255.0
MAX_OUT_VAL = 9.0
IMAGE_WIDTH = 28

class data_model:
    def __init__(self, X_train, y_train, X_assessment_train, y_assessment_train, X_assessment_test, y_assessment_test, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_assessment_train = X_assessment_train
        self.y_assessment_train = y_assessment_train
        self.X_assessment_test = X_assessment_test
        self.y_assessment_test = y_assessment_test
        self.X_test = X_test
        
def train_data_in_out():
    train_data = pd.read_csv('C:/Projects/Python/Kaggle/DigitsRecognizer/data_files/train.csv')
    train_data_in = train_data.iloc[:, 1:].to_numpy()
    train_data_out = train_data.iloc[:, :1].to_numpy()
    
    return train_data_in, train_data_out

def assessment_data_in_out(X_train_data, y_train_data):
    return train_test_split(
        X_train_data, 
        y_train_data, 
        test_size=0.2)

def test_data_in():
    test_data_in = pd.read_csv('C:/Projects/Python/Kaggle/DigitsRecognizer/data_files/test.csv')
    test_data_in = test_data_in.to_numpy()
    
    return test_data_in

def load_data():
    X_train, y_train = train_data_in_out()
    X_assessment_train, X_assessment_test, y_assessment_train, y_assessment_test = assessment_data_in_out(X_train, y_train)
    X_test = test_data_in()
    
    data = data_model(
        X_train=X_train, 
        y_train=y_train, 
        X_assessment_train=X_assessment_train, 
        y_assessment_train=y_assessment_train, 
        X_assessment_test=X_assessment_test, 
        y_assessment_test=y_assessment_test, 
        X_test=X_test)
    
    return data

def count_score(expected, actual):
    correct = 0
    wrong = 0
    
    for i in range(expected.shape[0]):
        if(expected[i] == actual[i]):
            correct += 1
        else:
            wrong += 1
            
    return correct / (correct + wrong)

def assess_epochs(model, data: data_model, epochs):
    model.learn(data.X_assessment_train, data.y_assessment_train, epochs=epochs)

    predictions = model.predict(data.X_assessment_test)

    score = count_score(predictions, data.y_assessment_test)
    print("epochs: " + str(epochs) + " score: " + str(score))
    
def assess_kernel_size(model, data: data_model, kernel_size):
    model.learn(data.X_assessment_train, data.y_assessment_train, kernel_size=kernel_size)

    predictions = model.predict(data.X_assessment_test)

    score = count_score(predictions, data.y_assessment_test)
    print("kernel_size: " + str(kernel_size) + " score: " + str(score))
    
def assess_filters(model, data: data_model, filters):
    model.learn(data.X_assessment_train, data.y_assessment_train, filters=filters)

    predictions = model.predict(data.X_assessment_test)

    score = count_score(predictions, data.y_assessment_test)
    print("filters: " + str(filters) + " score: " + str(score))
    
def assess_dense_layer_size(model, data: data_model, dense_layer_size):
    model.learn(data.X_assessment_train, data.y_assessment_train, dense_layer_size=dense_layer_size)

    predictions = model.predict(data.X_assessment_test)

    score = count_score(predictions, data.y_assessment_test)
    print("dense_layer_size: " + str(dense_layer_size) + " score: " + str(score))

def normalize_input(x):
    return x / MAX_IN_VAL

def normalize_output(x):
    return x / MAX_OUT_VAL

def inverse_normalize_input(x):
    return np.rint(x * MAX_IN_VAL)

def inverse_normalize_output(x):
    return np.rint(x * MAX_OUT_VAL)

def one_hot_encode(x):
    return OneHotEncoder().fit_transform(x).toarray()

def inverse_one_hot_encode(x):
    one_hot_decode_func = lambda y : y.argmax()
    decoded = map(one_hot_decode_func, x)
    return np.fromiter(decoded, dtype=int)

def two_dim_input(x):
    to_two_dim_func = lambda y : np.array_split(y, IMAGE_WIDTH)
    return list(map(to_two_dim_func, x))

class model_convolutional_normalized_in_out:
    def preprocess_in(data_in):
        preprocessed_in = normalize_input(data_in)
        preprocessed_in = two_dim_input(preprocessed_in)
        preprocessed_in = tensorflow.expand_dims(preprocessed_in, 3)
        
        return preprocessed_in

    def learn(self, data_in, data_out, kernel_size=12, filters=14, epochs=200, dense_layer_size=128):
        preprocessed_in = model_convolutional_normalized_in_out.preprocess_in(data_in)
        preprocessed_out = normalize_output(data_out)
        preprocessed_out = one_hot_encode(preprocessed_out)

        # get_ipython().push({'layer': layer})
        
        self.model = keras.Sequential()
        self.model.add(layers.Conv2D(kernel_size=kernel_size, filters=filters, activation="relu", input_shape=(IMAGE_WIDTH, IMAGE_WIDTH, 1)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(10, activation="softmax"))
        
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        
        self.model.fit(x=preprocessed_in, y=preprocessed_out, epochs=epochs)

    def predict(self, data_in):
        preprocessed_in = model_convolutional_normalized_in_out.preprocess_in(data_in)
        get_ipython().push({'preprocessed_in': preprocessed_in})

        predictions = self.model.predict(x=preprocessed_in)
        predictions = inverse_normalize_output(predictions)
        predictions = inverse_one_hot_encode(predictions)
        
        return predictions
    
def show(image_row):
    squared_image_row = np.array_split(image_row, IMAGE_WIDTH)
    plt.imshow(squared_image_row)

data = load_data()
model = model_convolutional_normalized_in_out()

# for i in range(1, 2):
#     assess_epochs(model, data, i)
    
model.learn(data.X_assessment_train, data.y_assessment_train)

assessment_results = model.predict(data.X_assessment_test)

score = count_score(data.y_assessment_test, assessment_results)
print(f"assessment score: {score}")


model.learn(data.X_train, data.y_train)
results = model.predict(data.X_train)

score = count_score(data.y_train, results)
print(f"training score: {score}")

final_results = model.predict(data.X_test)

with open('data_files/result.csv', 'w') as f:
    print("ImageId,Label", file=f)
    for i in range(len(final_results)):
        print(f"{i+1},{final_results[i]}", file=f)