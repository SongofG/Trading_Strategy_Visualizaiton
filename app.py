# This is the actual application python file.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.visualization import Visualizers
from utils.preprocess import Preprocess
from utils.model import LSTM


# Set Page Configuration
st.set_page_config(page_title="Stock Trading Strategies Visualization", page_icon="📈", layout="wide")


# TODO: Wrap these into an initializer
# Initialize the Session States
if 'ticker' not in st.session_state:
    st.session_state['ticker'] = 'AAPL'
if 'period' not in st.session_state:
    st.session_state['period'] = '2y'
if 'price_type' not in st.session_state:
    st.session_state['price_type'] = 'Close'
if 'ma_1' not in st.session_state:
    st.session_state['ma_1'] = 224
if 'window_size' not in st.session_state:
    st.session_state['window_size'] = 20
if 'train_ratio' not in st.session_state or st.session_state['train_ratio'] == '':
    st.session_state['train_ratio'] = '0.8'
if 'num_hidden_layers' not in st.session_state or st.session_state['num_hidden_layers'] == '':
    st.session_state['num_hidden_layers'] = 1
if 'learning_rate' not in st.session_state or st.session_state['learning_rate'] == '':
    st.session_state['learning_rate'] = 0.001
if 'epochs' not in st.session_state or st.session_state['epochs'] == '':
    st.session_state['epochs'] = 100
if 'lstm_button_clicked' not in st.session_state:
    st.session_state['lstm_button_clicked'] = [False] * 4  # Assuming a maximum of 4 buttons for example
if 'training_needed' not in st.session_state:
    st.session_state['training_needed'] = True
if 'lstm_model' not in st.session_state:
    st.session_state['lstm_model'] = None

# Function to set the state of the button
def set_lstm_button_state(index):
    st.session_state['lstm_button_clicked'][index] = True
    
def set_lstm_button_reset():
    st.session_state['lstm_button_clicked'] = [False] * 4
    st.session_state['training_needed'] = True
    st.session_state['lstm_model'] = None

# Function to call the callback functions and arguments
def call_function_by_index(functions_list, args_list, index):
    if index < len(functions_list) and index < len(args_list):
        functions_list[index](*args_list[index])
    else:
        st.warning("No Function for the button yet", icon="🔥")

# Tickers
tickers = pd.read_csv('data/tickers.csv')
sorted_tickers = sorted(tickers['ticker'].astype(str).unique())

# Title of the application
st.title("Stock Trading Strategies Visualization")

# Rough Guide
st.write("Please select the ticker of interst!")

ticker_choice, period_choice, price_choice = st.columns(3)

with ticker_choice:
    # Select Box: Ticker
    ticker = st.selectbox("What's the ticker of interest?", sorted_tickers, key='ticker', on_change=set_lstm_button_reset)

with period_choice:
    period = st.selectbox("What date period are you interested in?", ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'], key='period', on_change=set_lstm_button_reset)

with price_choice:
    # Select Box of the data to train for
    price_type = st.selectbox("What is your price type of interest to train the model?", ["Open", "Close", "High", "Low"], key="price_type", on_change=set_lstm_button_reset)

# Initialize the Visualizer Object
viz = Visualizers(ticker=st.session_state.ticker, period=st.session_state.period)

 # Candlesticks Chart Figure Object
fig = viz.historical_price_candlestick_chart()

st.markdown('#### Moving Average 1')
# Give Selectbox for the moving average period
st.slider("", min_value=30, max_value=365, key='ma_1')

viz.price_history['ma_1'] = viz.price_history['Close'].rolling(window=st.session_state['ma_1']).mean()
# viz.price_history['ma_2'] = viz.price_history['Close'].rolling(window=st.session_state['ma_2']).mean()

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

st.plotly_chart(fig, use_container_width=True)

# Trade Volume Chart
st.plotly_chart(viz.trade_volume_chart(), use_container_width=True)

# ACF and PACF Visualizations
acf_col, pacf_col = st.columns(2)

with acf_col:
    st.plotly_chart(viz.plot_acf_pacf(viz.price_history[price_type], nlags=30, alpha=0.05, is_acf=True), use_container_width=True)
with pacf_col:
    st.plotly_chart(viz.plot_acf_pacf(viz.price_history[price_type], nlags=30, alpha=0.05, is_acf=False), use_container_width=True)


# Tabs for each strategy
tab1, tab2 = st.tabs(["Strategy 1: LSTM", "Strategy 2: ARIMA"])

with tab1:
    
    st.header("LSTM")
    
    # Get the preprocessor ready
    preprocessor = Preprocess(viz.price_history, price_type)
    
    # Slider for the range of the window function
    window_size = st.slider("Give me the window_size!", min_value=0, max_value=viz.price_history[price_type].shape[0]//4, key='window_size', on_change=set_lstm_button_reset)  # limited the size of window function so that it cannot have the full range.
    
    # #Split the data into X and Y!
    # X, y = preprocessor.dataframe_to_X_y(viz.price_history[price_type], window_size=window_size)
    
    # Plot the target line
    line_chart = viz.line_chart(x=viz.price_history['Date'], y=viz.price_history[price_type], color='sky blue', width=1.5, xaxis_title='Date', yaxis_title='Price', title=f"{price_type} Price Over Date")
    st.plotly_chart(line_chart, use_container_width=True)
    
    # Get the ratio of training set from the user
    train_ratio = st.text_input("What ratio do you want your dataset to be a training set?", max_chars=4, key="train_ratio", on_change=set_lstm_button_reset)
    
    # Check if the ratio is acceptable data type
    try:
        train_ratio = float(train_ratio)
    except Exception as e:
        st.error('Training Ratio: Please enter float values!', icon="⚠️")
    
    # Get the number of hidden layers that user wants
    num_hidden_layers = st.number_input("How many hidden layers do you want for your model?", key='num_hidden_layers', min_value=1, max_value=10, step=1, on_change=set_lstm_button_reset)
    
    layer_config = []
    # Interactive layer config inputs
    for i in range(num_hidden_layers):
        st.write(f'##### Layer {i+1} Configuration')
        col1, col2 = st.columns(2)
        
        with col1:
            activation = st.selectbox(
                "Choose activation function",
                ['relu', 'sigmoid', 'tanh'],
                key=f"activation_{i+1}"
            )
        
        with col2:
            neurons = st.number_input(
                "Number of neurons",
                min_value=1,
                max_value=1024,
                step=1,
                key=f"neuron_{i+1}"
            )
        
        layer_config.append((activation, neurons))
        
    # Get the epoch
    epochs = st.number_input("What the number of epochs?", key='epochs', min_value=1, max_value=10000, step=1, on_change=set_lstm_button_reset)
    
    # Get the learning rate
    learning_rate = st.number_input("What is the learning rate?", key='learning_rate', min_value=0.001, max_value=1.0, step=0.001, format="%.3f", on_change=set_lstm_button_reset)
        
    # Preprocess the data
    dates, X, y = preprocessor.windowed_df_to_dates_X_y(n=window_size)
    result = preprocessor.train_validation_test_split(dates, X, y, train_ratio)
    
    # Get the model object
    if st.session_state['training_needed']:
        lstm = LSTM(input_shape=(window_size, 1), lstm_neuron_num=num_hidden_layers, layer_and_activation=layer_config, learning_rate=learning_rate, training_needed=st.session_state['training_needed'])
        st.session_state['lstm_model'] = lstm
    else:
        st.session_state['lstm_model'].set_training_needed(st.session_state['training_needed'])
        
    functions_list = [viz.plot_train_validation_split, viz.plot_train_validation_result, viz.plot_test_result, st.session_state['lstm_model'].save_model]  # Actual Model Fitting and Plotting
    args_list = [(result,), (st.session_state['lstm_model'], result, epochs), (st.session_state['lstm_model'], result), (st.session_state['ticker']+'_model',)]
        
    for i, s in enumerate([('Split the data!', 'split'), ('Train and Validate!', 'train'), ('Test!', 'test'), ('Save Your Model', 'save')]):
        if i == 0 or st.session_state['lstm_button_clicked'][i-1]:
            if not st.session_state['lstm_button_clicked'][i]:
                button = st.button(f'{s[0]}', key=f'{s[1]}', on_click=set_lstm_button_state, args=(i,))
                if button:
                    call_function_by_index(functions_list=functions_list, args_list=args_list, index=i)
            else:
                call_function_by_index(functions_list=functions_list, args_list=args_list, index=i)

    # Reset button
    if st.button('Reset'):
        # Reset all buttons in the session state
        set_lstm_button_reset()

with tab2:
    st.header("ARIMA")
    