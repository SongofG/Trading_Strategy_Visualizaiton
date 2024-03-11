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
    st.session_state['period'] = '2y'
if 'ma_1' not in st.session_state:
    st.session_state['ma_1'] = 224
if 'ma_2' not in st.session_state:
    st.session_state['ma_2'] = 112

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

# # Candlestick Chart of the target ticker
# st.plotly_chart(viz.historical_price_candlestick_chart(), use_container_width=True)

# Tabs for each strategy
tab1, tab2, tab3 = st.tabs(["Strategy 1: Moving Average", "Strategy 2: ARIMA", "Strategy 3: LSTM"])

with tab1:
    st.header("Moving Average")
    
    # Candlesticks Chart Figure Object
    fig = viz.historical_price_candlestick_chart()
   
    # Add columns for the moving average sliders
    col1, col2 = st.columns(2)
   
    # MA 1 column
    with col1:
        st.markdown('#### Moving Average 1')
        # Give Selectbox for the moving average period
        st.slider("", min_value=30, max_value=365, value=1, key='ma_1')
    with col2:
        st.markdown('#### Moving Average 2')
        # Give textinput box
        st.slider("", min_value=30, max_value=365, value=1, key='ma_2')
   
    viz.price_history['ma_1'] = viz.price_history['Close'].rolling(window=st.session_state['ma_1']).mean()
    viz.price_history['ma_2'] = viz.price_history['Close'].rolling(window=st.session_state['ma_2']).mean()
   
    # Add MA_1 Line
    fig.add_trace(
        go.Scatter(
            x=viz.price_history['Date'],
            y=viz.price_history['ma_1'],
            mode='lines',
            line={"width": 1.25},
            name="MA 1",
            opacity=0.45,
            marker={'color': 'black'}
        )
    )
    
    # Add MA_2 Line
    fig.add_trace(
        go.Scatter(
            x=viz.price_history['Date'],
            y=viz.price_history['ma_2'],
            mode='lines',
            line={"width": 1.25},
            name="MA 2",
            opacity=0.45,
            marker={'color': 'orange'}
        )
    )
   
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade Volume Chart
    st.plotly_chart(viz.trade_volume_chart(), use_container_width=True)
   

with tab2:
    st.header("ARIMA")

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
    st.header("LSTM")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=price3_df['Date'], y=price3_df['Price'], mode='lines', name='Price'))

    # Update layout
    fig.update_layout(
        title='Random Stock Price Chart',
        xaxis_title='Date',
        yaxis_title='Price'
    )

    st.plotly_chart(fig, use_container_width=True)
