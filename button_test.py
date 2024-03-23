import streamlit as st

st.title("Interactive Button Generation Test")

# Initialize session states for buttons
if 'button_clicked' not in st.session_state:
    st.session_state['button_clicked'] = [False] * 10  # Assuming a maximum of 10 buttons for example

# Function to set the state of the button
def set_button_state(index):
    st.session_state['button_clicked'][index] = True

# Create buttons based on the previous button clicks
for i in range(10):
    if i == 0 or st.session_state['button_clicked'][i-1]:  # The first button always shows, subsequent ones depend on the previous button being clicked
        if not st.session_state['button_clicked'][i]:
            button = st.button(f'Button {i+1}', key=f'button{i+1}', on_click=set_button_state, args=(i,))
            if button:
                st.success(f'Button {i+1} clicked')
        else:
            st.success(f'Button {i+1} clicked')  # Display success message if already clicked

# Reset button
if st.button('Reset'):
    # Reset all buttons in the session state
    st.session_state['button_clicked'] = [False] * 10

# # Call the function to add buttons
# add_buttons(10)