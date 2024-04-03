import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import layers
import streamlit as st

class LSTM:
    
    def __init__(self, input_shape, lstm_neuron_num, layer_and_activation, learning_rate, training_needed):
        
        self._input_shape = input_shape
        self._lstm_neuron_num = lstm_neuron_num
        self._layer_and_activation = layer_and_activation  # A List of tuples of number of neurons and activation functions
        self._learning_rate = learning_rate
        self.training_needed = training_needed 
    
    
    def _create_model(self):
        
        # Get the model layers
        self.layers_list = [layers.Input(self._input_shape)] + [layers.LSTM(self._lstm_neuron_num)] + [layers.Dense(tup[1], activation=tup[0]) for tup in self._layer_and_activation] + [layers.Dense(1)]
        
        # Get the model
        self.model = Sequential(self.layers_list)
        
        # Compile the model
        self.model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self._learning_rate),
            metrics=['mean_absolute_error']
        )    
    
    
    def fit(self, X_train, y_train, X_validation, y_validation, epochs):
        
        self._create_model()
        
        # Check if the data types are acceptable
        if type(X_train) is not tf.float32:
            X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        if type(y_train) is not tf.float32:
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        if type(X_validation) is not tf.float32:
            X_validation = tf.convert_to_tensor(X_validation, dtype=tf.float32)
        if type(y_validation) is not tf.float32:
            y_validation = tf.convert_to_tensor(y_validation, dtype=tf.float32)
        
        # if self.training_needed:
        self.model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=epochs)
        
    
    def predict(self, X):
        
        # Check if the data type is acceptable
        if type(X) is not tf.float32:
            X = tf.convert_to_tensor(X, dtype=tf.float32)
        
        return self.model.predict(X).flatten()
    
    
    def set_training_needed(self, bool):
        self.training_needed = bool
    
    
    def save_model(self, filepath):
        self.model.save('my_models/' + filepath)
        st.success("Model Saved!")
    