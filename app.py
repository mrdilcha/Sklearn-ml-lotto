import streamlit as st
import pandas as pd
import numpy as np
from collections import deque

# Function definitions (keeping only the necessary ones)
def historical_pattern_recognition(history, period):
    pattern_length = 3  # Look for repeating patterns of length 3
    if len(history) < pattern_length:
        return np.random.choice([0, 1])  # No sufficient history, random prediction
    pattern = history[-pattern_length:]  # Extract the last pattern
    for i in range(len(history) - pattern_length):
        if np.array_equal(history[i:i+pattern_length], pattern):
            return history[i+pattern_length]  # If a pattern is found, predict the next outcome
    return np.random.choice([0, 1])  # If no pattern is found, random prediction

def trend_following(history):
    if len(history) < 3:
        return np.random.choice([0, 1])
    return history[-1]  # Follow the most recent trend

def predict_outcome(period_number, history):
    pattern_prediction = historical_pattern_recognition(history, period_number)
    trend_prediction = trend_following(history)
    # Combine predictions (you can adjust the weighting if needed)
    return 1 if (pattern_prediction + trend_prediction) > 1 else 0

# Add a title to your app
st.title("Period Number Prediction Application")

# Load dataset
uploaded_file = st.file_uploader("Upload a CSV file with historical data", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write("Here are the first few rows of your historical data:")
    st.write(data.head())
    
    # Feature and label extraction
    X = data['period_number'].values
    y = data['outcome'].values  # 0: Small, 1: Big
    
    # User input for prediction
    st.write("Enter the last three digits of the period number you want to predict:")
    user_input = st.text_input("Last three digits of period number:", max_chars=3)
    
    if st.button("Predict") and user_input:
        try:
            input_number = int(user_input)
            if 0 <= input_number <= 999:
                # Prepare history for prediction
                history = deque(y[-10:], maxlen=10)  # Use last 10 outcomes as history
                
                # Make prediction
                prediction = predict_outcome(input_number, list(history))
                
                # Display prediction
                st.write(f"Prediction for period number ending with {user_input}:")
                if prediction == 1:
                    st.write("The predicted outcome is: **Big**")
                else:
                    st.write("The predicted outcome is: **Small**")
                
                # Display some additional information
                st.write("\nRecent historical data:")
                recent_data = data.tail(10)
                st.write(recent_data)
                
                st.write("\nNote: This prediction is based on historical patterns and recent trends. "
                         "It should not be considered as financial advice or a guarantee of future outcomes.")
            else:
                st.write("Please enter a number between 000 and 999.")
        except ValueError:
            st.write("Please enter a valid three-digit number.")
else:
    st.write("Please upload a CSV file with historical data to continue.")
