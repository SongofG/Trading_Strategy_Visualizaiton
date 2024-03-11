import pandas as pd
import numpy as np

class Preprocess:
    
    def __init__(self):
        
        pass
    
    def dataframe_to_X_y(self, df, window_size=5):
        
        # convert the dataframe into numpy
        df_as_np = df.to_numpy()
        X = []
        y = []
        
        # Iterate over the numpy object
        for i in range(len(df_as_np)-window_size):
            row = [[a] for a in df_as_np[i:i+window_size]]
            X.append(row)
            label = df_as_np[i+5]
            y.append(label)
            
        return np.array(X), np.array(y)

