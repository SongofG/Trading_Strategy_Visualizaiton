# This is the actual application python file.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.visualization import Visualizers
import pickle


# Set Page Configuration
st.set_page_config(page_title="Stock Trading Strategies Visualization", page_icon="📈", layout="wide")

# TODO: Wrap these into an initializer
# Initialize the Session States
if 'ticker' not in st.session_state:
    st.session_state['ticker'] = 'AAPL'

if 'period' not in st.session_state:
    st.session_state['period'] = '6mo'

# Random Stock price generating data
# Generate random data for the stock price
# np.random.seed(0)
dates = pd.date_range('2023-01-01', '2023-12-31')
price1 = np.cumsum(np.random.randn(len(dates))) + 100
price2 = np.cumsum(np.random.randn(len(dates))) + 100
price3 = np.cumsum(np.random.randn(len(dates))) + 100

price1_df = pd.DataFrame({'Date': dates, 'Price': price1})
price2_df = pd.DataFrame({'Date': dates, 'Price': price2})
price3_df = pd.DataFrame({'Date': dates, 'Price': price3})

# Tickers
tickers = pd.read_csv('data/tickers.csv')
sorted_tickers = sorted(tickers['ticker'].astype(str).unique())

# Title of the application
st.title("Stock Trading Strategies Visualization")

# Rough Guide
st.write("Please select the ticker of interst!")

# Select Box: Ticker
ticker = st.selectbox("What's the ticker of interest?", sorted_tickers, key='ticker')

period = st.selectbox("What date period are you interested in?", ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'], key='period')

# Initialize the Visualizer Object
viz = Visualizers(ticker=st.session_state.ticker, period=st.session_state.period)

# Candlestick Chart of the target ticker
st.plotly_chart(viz.historical_price_candlestick_chart(), use_container_width=True)

# Trade Volume Chart
st.plotly_chart(viz.trade_volume_chart(), use_container_width=True)

# Tabs for each strategy
tab1, tab2, tab3 = st.tabs(["Strategy 1: Moving Average", "Strategy 2", "Strategy 3"])

with tab1:
   st.header("Strategy 1: Moving Average")
   
   # Give Selectbox for the moving average period
   # Sidebar for user input
   ma_period = st.slider('MA 1', min_value=30, max_value=120, value=1)
   
   viz.price_history['ma1'] = viz.price_history['Close'].rolling(window=ma_period).mean()
   
   st.write(viz.price_history.head())
#    # Sidebar for user input
#    period = st.slider('MA 2', min_value=30, max_value=120, value=1)
#    # Sidebar for user input
#    period = st.slider('MA 3', min_value=30, max_value=120, value=1)
   
   st.plotly_chart(viz.historical_price_candlestick_chart(), use_container_width=True)
   

with tab2:
   st.header("Strategy 2")
   
   fig = go.Figure()
   
   fig.add_trace(go.Scatter(x=price2_df['Date'], y=price2_df['Price'], mode='lines', name='Price'))
   # Update layout
   fig.update_layout(
       title='Random Stock Price Chart',
       xaxis_title='Date',
       yaxis_title='Price'
   )

   st.plotly_chart(fig, use_container_width=True)
   

with tab3:
   st.header("Strategy 3")
   
   fig = go.Figure()
   
   fig.add_trace(go.Scatter(x=price3_df['Date'], y=price3_df['Price'], mode='lines', name='Price'))
   
   # Update layout
   fig.update_layout(
       title='Random Stock Price Chart',
       xaxis_title='Date',
       yaxis_title='Price'
   )

   st.plotly_chart(fig, use_container_width=True)
