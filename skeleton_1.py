import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# tickers
tickers = pd.read_csv('data/tickers.csv')

# Random Stock price generating data
# Generate random data for the stock price
np.random.seed(0)
dates = pd.date_range('2023-01-01', '2023-12-31')
price1 = np.cumsum(np.random.randn(len(dates))) + 100
price2 = np.cumsum(np.random.randn(len(dates))) + 100
price3 = np.cumsum(np.random.randn(len(dates))) + 100

price1_df = pd.DataFrame({'Date': dates, 'Price': price1})
price2_df = pd.DataFrame({'Date': dates, 'Price': price2})
price3_df = pd.DataFrame({'Date': dates, 'Price': price3})

# Define functions for each page
def home():
    
    st.title("Stock Trading Strategies Visualization")
    st.write("Welcome to the Home page!")
    
    st.write("Recommended list of stocks")
    
    st.write(tickers.head())
    
    

def strategy_1():
    st.title("Strategy 1: Something")
    
    fig = go.Figure()
   
    fig.add_trace(go.Scatter(x=price1_df['Date'], y=price1_df['Price'], mode='lines', name='Price'))
    
    # Update layout
    fig.update_layout(
       title='Random Stock Price Chart',
       xaxis_title='Date',
       yaxis_title='Price'
    )

    st.plotly_chart(fig)
    
    
def strategy_2():
    st.title("Strategy 2: Something")
    
    
    fig = go.Figure()
   
    fig.add_trace(go.Scatter(x=price2_df['Date'], y=price2_df['Price'], mode='lines', name='Price'))
    
    # Update layout
    fig.update_layout(
       title='Random Stock Price Chart',
       xaxis_title='Date',
       yaxis_title='Price'
    )

    st.plotly_chart(fig)
    
    
def strategy_3():
    st.title("Strategy 3: Something")
    
    
    fig = go.Figure()
   
    fig.add_trace(go.Scatter(x=price3_df['Date'], y=price3_df['Price'], mode='lines', name='Price'))
    
    # Update layout
    fig.update_layout(
       title='Random Stock Price Chart',
       xaxis_title='Date',
       yaxis_title='Price'
    )

    st.plotly_chart(fig)

# Create a sidebar navigation menu
page = st.sidebar.selectbox("Select a page", ["Home", "Strategy 1", "Strategy 2", "Strategy 3"])

# Display the selected page based on user input
if page == "Home":
    home()
elif page == "Strategy 1":
    strategy_1()
elif page == "Strategy 2":
    strategy_2()
elif page == "Strategy 3":
    strategy_3()