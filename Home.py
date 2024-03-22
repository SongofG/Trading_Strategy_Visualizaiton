# This is the actual application python file.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.visualization import Visualizers
from utils.preprocess import Preprocess


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
if 'window_size' not in st.session_state:
    st.session_state['window_size'] = 20
if 'train_ratio' not in st.session_state or st.session_state['train_ratio'] == '':
    st.session_state['train_ratio'] = '0.8'

# Random Stock price generating data
# Generate random data for the stock price
# np.random.seed(0)
dates = pd.date_range('2023-01-01', '2023-12-31')
price2 = np.cumsum(np.random.randn(len(dates))) + 100
price2_df = pd.DataFrame({'Date': dates, 'Price': price2})

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


# Tabs for each strategy
tab1, tab2 = st.tabs(["Strategy 1: LSTM", "Strategy 2: ARIMA"])

with tab1:
    
    st.header("LSTM")
    
    # Select Box of the data to train for
    price_type = st.selectbox("What is your price type of interest to train the model?", ["Open", "Close", "High", "Low"], key="price_type")
    
    # Get the preprocessor ready
    preprocessor = Preprocess(viz.price_history, price_type)
    
    # Slider for the range of the window function
    window_size = st.slider("Give me the window_size!", min_value=0, max_value=viz.price_history[price_type].shape[0]//4, key='window_size')  # limited the size of window function so that it cannot have the full range.
    
    # #Split the data into X and Y!
    # X, y = preprocessor.dataframe_to_X_y(viz.price_history[price_type], window_size=window_size)
    
    # Plot the target line
    line_chart = viz.line_chart(x=viz.price_history['Date'], y=viz.price_history[price_type], color='sky blue', width=1.5, xaxis_title='Date', yaxis_title='Price', title=f"{price_type} Price Over Date")
    st.plotly_chart(line_chart, use_container_width=True)
    
    dates, X, y = preprocessor.windowed_df_to_dates_X_y(n=window_size)
    
    # Get the ratio of training set from the user
    train_ratio = st.text_input("What ratio do you want your dataset to be a training set?", max_chars=4, key="train_ratio")
    
    # Check if the ratio is acceptable data type
    try:
        train_ratio = float(train_ratio)
    except Exception as e:
        st.error('Please enter float values!', icon="⚠️")
    
    split_data = st.button("Split the Dataset!")
        
    # Split into train, validation, and test and visualize them.
    if split_data:
        result = preprocessor.train_validation_test_split(dates, X, y, train_ratio)
        
        # Visualize the train, validaiton, and split
        train_validation_test_split_line = viz.train_validation_test_visualization(result)
        
        st.write("Here is how your data is splited!")
        st.plotly_chart(train_validation_test_split_line, use_container_width=True)
   

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
    