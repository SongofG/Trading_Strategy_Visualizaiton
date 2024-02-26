import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from yfinance.base import TickerBase

def historical_price_candlestick_chart(ticker, period):
    
    """
    :Parameters:
    
        ticker : str
                Ticker of the target company
        period : str
                Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                Either Use period parameter or use start and end
    """
    
    # Get the target TickerBase Object
    target = TickerBase(ticker)
    
    # Get the price history in DataFrame
    price_history = target.history(period).reset_index()
    
    # Parse the date column to be datetime
    price_history['Date'] = pd.to_datetime(price_history['Date'])
    
    fig = go.Figure(data=[
        go.Candlestick(
            x=price_history['Date'],
            open=price_history['Open'],
            high=price_history['High'],
            low=price_history['Low'],
            close=price_history['Close']
        )
    ])
    
    fig.update_layout(
        title=f'Candlestick Chart: {ticker}',
        xaxis_title='Date',
        yaxis_title='Price'
    )
    
    st.plotly_chart(fig)