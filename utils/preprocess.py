import pandas as pd
import numpy as np
import datetime

class Preprocess:
    
    def __init__(self, df, target):
        
        self.df = df[['Date', target]]
        self.target = target
        
    
    def _str_to_datetime(self, s):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.date(year=year, month=month, day=day)
    
    def _date_to_index(self):
        """
            This method pops the Date column to be the index of the df.
        """
        self.df['Date'] = self.df['Date'].dt.date
    
    
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
    
    
    def train_validation_test_split(X, y, train_ratio):
        
        validation_ratio = round((1-train_ratio)/2, 2)
        
        train_q = int(len(X) * train_ratio)
        validation_q = int(len(X) * (train_ratio + validation_ratio))
        
        # Split X
        X_train, X_validation, X_test = X[:train_q], X[train_q:validation_q], X[validation_q:]
        
        # Split y
        y_train, y_validation, y_test = y[:train_q], y[train_q:validation_q], y[validation_q:]
        
        return X_train, X_validation, X_test, y_train, y_validation, y_test
