import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from yfinance.base import TickerBase

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
            showlegend=False
        )
        
        return fig
