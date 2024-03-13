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
        
    
    def df_to_windowed_df(self, dataframe, n=3):
        
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

            values = df_subset['Close'].to_numpy()
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
    
    
    def windowed_df_to_dates_X_y(self, windowed_df):
        
        df_as_np = windowed_df.to_numpy()
        
        dates = df_as_np[:, 0]

        middle_matrix = df_as_np[:, 1:-1]
        X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

        y = df_as_np[:, -1]
        
        return dates, X, y
    
    
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
