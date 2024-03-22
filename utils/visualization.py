import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from yfinance.base import TickerBase
from statsmodels.tsa.stattools import acf, pacf

class Visualizers():
    
    def __init__(self, ticker, period):
        """
        :Parameters:
        
            ticker : str
                    Ticker of the target company
            period : str
                    Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                    Either Use period parameter or use start and end
        """
        
        self.ticker = ticker
        self.period = period
        self.target = TickerBase(ticker)
        
        # Get the Date column instead of the date index
        self.price_history = self.target.history(period).reset_index()
        
        # Parse the date column to be datetime
        self.price_history['Date'] = pd.to_datetime(self.price_history['Date'])
        
        # Add the "Close_Higher_Today_Flag"
        self.price_history['Close_Higher_Today'] = self.price_history['Close'] > self.price_history['Close'].shift(1)


    def historical_price_candlestick_chart(self):
        """
            This function visualize candlesticks chart of the given ticker and period
        """
        
        
        fig = go.Figure(data=[
            go.Candlestick(
                x=self.price_history['Date'],
                open=self.price_history['Open'],
                high=self.price_history['High'],
                low=self.price_history['Low'],
                close=self.price_history['Close']
            )
        ])
        
        fig.update_layout(
            title=f'Candlestick Chart: {self.ticker}',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
        )
        
        return fig
        
    
    def trade_volume_chart(self):
        
        # Color map
        color_map = {True: '#3D9970', False: '#FF4136'}
        
        # Bar chart
        fig = px.bar(self.price_history, x='Date', y='Volume', color='Close_Higher_Today', color_discrete_map=color_map)
        
        # Update the layout
        fig.update_layout(
            title=f'Trade Volume Chart: {self.ticker}',
            xaxis_title='Date',
            yaxis_title='Volume',
            # width=1550,
            # height=600,
            barcornerradius=15,
            # showlegend=False
        )
        
        return fig
    
    def line_chart(self, title, x, y, color, width, xaxis_title, yaxis_title):
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=color, width=width)))
        
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title
        )
        
        return fig

    
    def train_validation_test_visualization(self, results):
        
        # We unpact the train, validation, and test for the data
        dates = results['dates']
        y = results['y']
        
        # Get the lines
        train_trace = go.Scatter(x=dates['train'], y=y['train'], mode='lines', name='Train', line=dict(color='sky blue'))
        validation_trace = go.Scatter(x=dates['validation'], y=y['validation'], mode='lines', name='Validation', line=dict(color='orange'))
        test_trace = go.Scatter(x=dates['test'], y=y['test'], mode='lines', name='Test', line=dict(color='green'))
        
        fig = go.Figure(data=[train_trace, validation_trace, test_trace])
        
        # Customize layout
        fig.update_layout(
            title='Train, Validaiton, and Test Split!',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Dataset',
            legend=dict(
            x=0,
            y=0.75,
            xanchor='left',
            yanchor='bottom'
                )
            )
        
        return fig
    
    
    def plot_acf_pacf(self, data, nlags, alpha, is_acf=True):
        
        # Get the ACF and Confidence Intervals
        lags = np.arange(0, int(nlags) + 1)  # +1 for zero lag
        
        # Check if it's ACF or PACF
        if is_acf:
            cf_x = acf(data, nlags=nlags, alpha=alpha)
            title = 'Autocorrelation Function (ACF)'
        else:
            cf_x = pacf(data, nlags=nlags, alpha=alpha)
            title = 'Partially Autocorrelation Function (PACF)'
        
        CF, confint = cf_x[:2]
        
        # Calculate the Lower and Upper Bounds
        lower_bound = confint[:, 0] - CF
        upper_bound = confint[:, 1] - CF
        
        # Upper & Lower bounds
        
        upper_bound_trace = go.Scatter(
            x = lags,
            y = upper_bound,
            mode='lines',
            fill=None,
            line=dict(color='rgba(0,100,80,0.2)'),
            name='Upper Bound',
            showlegend=False
        )

        lower_bound_trace = go.Scatter(
            x = lags,
            y = lower_bound,
            mode='lines',
            fill='tonexty',  # This fills the area between this trace and the one above it
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(0,100,80,0.2)'),
            name='Lower Bound',
            showlegend=False
        )
        
        # Plot ACFs in Vertical Lines
        cf_lines = []
        for lag in lags:
            
            line = go.Scatter(
                x=[lag, lag],
                y=[0,CF[lag]] if CF[lag] > 0 else [CF[lag], 0],
                mode='lines+markers',
                line=dict(color='orange'),
                showlegend=False
            )
            
            cf_lines.append(line)
        
        fig = go.Figure(data=cf_lines + [upper_bound_trace, lower_bound_trace])
        
        fig.update_layout(
            title=title,
            yaxis=dict(range=[-1.2, 1.2])
        )
        
        return fig
    