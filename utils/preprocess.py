import pandas as pd
import numpy as np
import datetime
import streamlit as st
import tensorflow as tf

class Preprocess:
    
    def __init__(self, df, target):
        
        self.df = df[['Date', target]]
        self.target = target
        
    
    def _date_to_index(self):
        """
            This method pops the Date column to be the index of the df.
        """
        
        # Cast the type of the Date column
        self.df['Date'] = self.df['Date'].dt.date
        
        # Let Dates be index
        self.df.index = self.df.pop('Date')
        
    
    def _df_to_windowed_df(self, dataframe, n=3):
        
        first_date = dataframe.index[n]  # The minimum date from that's n rows away from the start
        last_date  = dataframe.index[-1]  # The last date

        target_date = first_date
        
        dates = []
        X, Y = [], []

        last_time = False
        while True:
            df_subset = dataframe.loc[:target_date].tail(n+1)
            
            if len(df_subset) != n+1:
                print(f'Error: Window of size {n} is too large for date {target_date}')
                return

            values = df_subset[self.target].to_numpy()
            x, y = values[:-1], values[-1]

            dates.append(target_date)
            X.append(x)
            Y.append(y)

            next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
            next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
            next_date_str = next_datetime_str.split('T')[0]
            year_month_day = next_date_str.split('-')
            year, month, day = year_month_day
            next_date = datetime.date(day=int(day), month=int(month), year=int(year))
            
            if last_time:
                break
            
            target_date = next_date

            if target_date == last_date:
                last_time = True
            
        ret_df = pd.DataFrame({})
        ret_df['Target Date'] = dates
        
        X = np.array(X)
        for i in range(0, n):
            X[:, i]
            ret_df[f'Target-{n-i}'] = X[:, i]
        
        ret_df['Target'] = Y

        return ret_df
    
    
    def windowed_df_to_dates_X_y(self, n=3):
        
        self._date_to_index()
        
        windowed_df = self._df_to_windowed_df(self.df, n)
        
        df_as_np = windowed_df.to_numpy()
        
        dates = df_as_np[:, 0]

        middle_matrix = df_as_np[:, 1:-1]
        X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

        y = df_as_np[:, -1]
        
        return dates, X, y
    
    
    # Modify this fuction accordingly to the above functions
    def train_validation_test_split(self, dates, X, y, train_ratio):
        
        # Check if the given ratio is acceptable or not.
        if train_ratio <= 0 or train_ratio > 1:
            st.error('The ratio you have given is not in valid range', icon='🚨')
            return
        
        validation_ratio = round((1-train_ratio)/2, 2)
        
        train_q = int(len(X) * train_ratio)
        validation_q = int(len(X) * (train_ratio + validation_ratio))
        
        # Split dates
        dates_train, dates_validation, dates_test = dates[:train_q], dates[train_q:validation_q], dates[validation_q:]
        
        # Split X, and type casting
        X_train, X_validation, X_test = tf.convert_to_tensor(X[:train_q], dtype=tf.float32), tf.convert_to_tensor(X[train_q:validation_q], dtype=tf.float32), tf.convert_to_tensor(X[validation_q:], dtype=tf.float32)
        
        # Split y and type casting
        y_train, y_validation, y_test = tf.convert_to_tensor(y[:train_q], dtype=tf.float32), tf.convert_to_tensor(y[train_q:validation_q], dtype=tf.float32), tf.convert_to_tensor(y[validation_q:], dtype=tf.float32)
        
        # Package the outputs
        results = {
            "X": {"train": X_train, "validation": X_validation, "test": X_test},
            "y": {"train": y_train, "validation": y_validation, "test": y_test},
            "dates": {"train": dates_train, "validation": dates_validation, "test": dates_test}
        }
        
        return results
