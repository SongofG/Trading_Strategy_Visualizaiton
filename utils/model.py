from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import layers

class LSTM:
    
    def __init__(self, input_shape, lstm_neuron_num, layer_and_activation, learning_rate):
        
        self._input_shape = input_shape
        self._lstm_neuron_num = lstm_neuron_num
        self._layer_and_activation = layer_and_activation  # A List of tuples of number of neurons and activation functions
        self._learning_rate = learning_rate
        
        # Get the model layers
        layers_list = [layers.input(self._input_shape)] 
        + [layers.LSTM(self._lstm_neuron_num)] 
        + [layers.Dense(tup[0], activation=tup[1]) for tup in self._layer_and_activation]
        + [layers.Dense(1)]
        
        # Get the model
        self.model = Sequential(layers_list)
        
        # Compile the model
        self.model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self._learning_rate),
            metrics=['mean_absolute_error']
        )    
    
    
    def fit(self, X_train, y_train, X_validation, y_validation, epoch):
        self.model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epoch=epoch)
        
    
    def predict(self, X):
        return self.model.predict(X).flatten()
    