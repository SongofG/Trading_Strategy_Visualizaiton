# This is the actual application python file.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.visualization import Visualizers


# Set Page Configuration
st.set_page_config(page_title="Stock Trading Strategies Visualization", page_icon="📈", layout="wide")

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

# Select Box
ticker = st.selectbox("What's the ticker of interest?", sorted_tickers)
period = st.selectbox("What date period are you interested in?", ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'])

# Initialize the Visualizer Object
viz = Visualizers(ticker=ticker, period=period)

# Candlestick Chart of the target ticker
viz.historical_price_candlestick_chart()

# Trade Volume Chart
viz.trade_volume_chart()

# # Tabs for each strategy
# tab1, tab2, tab3 = st.tabs(["Strategy 1", "Strategy 2", "Strategy 3"])

# with tab1:
#    st.header("Strategy 1")
   
#    fig = go.Figure()
   
#    fig.add_trace(go.Scatter(x=price1_df['Date'], y=price1_df['Price'], mode='lines', name='Price'))
#    # Update layout
#    fig.update_layout(
#        title='Random Stock Price Chart',
#        xaxis_title='Date',
#        yaxis_title='Price'
#    )

#    st.plotly_chart(fig)
   

# with tab2:
#    st.header("Strategy 2")
   
#    fig = go.Figure()
   
#    fig.add_trace(go.Scatter(x=price2_df['Date'], y=price2_df['Price'], mode='lines', name='Price'))
#    # Update layout
#    fig.update_layout(
#        title='Random Stock Price Chart',
#        xaxis_title='Date',
#        yaxis_title='Price'
#    )

#    st.plotly_chart(fig)
   

# with tab3:
#    st.header("Strategy 3")
   
#    fig = go.Figure()
   
#    fig.add_trace(go.Scatter(x=price3_df['Date'], y=price3_df['Price'], mode='lines', name='Price'))
   
#    # Update layout
#    fig.update_layout(
#        title='Random Stock Price Chart',
#        xaxis_title='Date',
#        yaxis_title='Price'
#    )

#    st.plotly_chart(fig)
