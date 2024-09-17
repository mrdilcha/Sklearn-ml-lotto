import streamlit as st 
import numpy as np
import pandas as pd
from collections import deque


# Add a title to your app
st.title("Prediction Application")

# Load dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write("Here is the first few rows of your data:")
    st.write(data.head())


# Load the historical data (Assuming it's already available in 'data.csv')
data = pd.read_csv('data.csv')

# Feature extraction
X = data['period_number'].values
y = data['outcome'].values  # 0: Small, 1: Big

# Function to calculate moving averages
def moving_average(arr, window_size=5):
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

# Historical Pattern Recognition (Search for patterns in the last few outcomes)
def historical_pattern_recognition(history, period):
    pattern_length = 5  # Look for repeating patterns of length 5
    if len(history) < pattern_length:
        return np.random.choice([0, 1])  # No sufficient history, random prediction
    pattern = history[-pattern_length:]  # Extract the last pattern
    for i in range(len(history) - pattern_length):
        if np.array_equal(history[i:i+pattern_length], pattern):
            return history[i+pattern_length]  # If a pattern is found, predict the next outcome
    return np.random.choice([0, 1])  # If no pattern is found, random prediction

# Moving Average (Predict based on average of last few outcomes)
def moving_average_prediction(history):
    ma = moving_average(history, window_size=3)
    if len(ma) == 0:
        return np.random.choice([0, 1])  # Not enough data for moving average
    return 1 if ma[-1] > 0.5 else 0

# Support and Resistance Levels (If there’s a repeating level, follow it)
def support_resistance_prediction(history):
    small_count = np.sum(np.array(history) == 0)
    big_count = np.sum(np.array(history) == 1)
    return 0 if small_count > big_count else 1  # Predict the majority outcome

# Trend Following (Predict based on the current trend)
def trend_following(history):
    if len(history) < 3:
        return np.random.choice([0, 1])
    return history[-1]  # Follow the most recent trend

# Mean Reversion (If too many consecutive "Big" or "Small" outcomes, predict the opposite)
def mean_reversion(history):
    consecutive = 3  # Threshold for mean reversion
    if len(history) >= consecutive and all(x == history[-1] for x in history[-consecutive:]):
        return 1 - history[-1]  # Predict the opposite
    return np.random.choice([0, 1])  # Random prediction otherwise

# Time Series Analysis (Using lagged values to predict the next outcome)
def time_series_analysis(history):
    lag = 2  # Use the last 2 outcomes as the lag
    if len(history) < lag:
        return np.random.choice([0, 1])
    return history[-lag]  # Predict based on the lag

# Momentum Indicators (Simple heuristic based on momentum)
def momentum_prediction(history):
    if len(history) < 3:
        return np.random.choice([0, 1])
    momentum = sum(history[-3:])  # Calculate the momentum (last 3 outcomes)
    return 1 if momentum > 1 else 0  # Predict based on momentum

# Rule-based Prediction (Simple heuristic rules)
def rule_based_prediction(history):
    if len(history) == 0:
        return np.random.choice([0, 1])
    # Rule: Predict the opposite if the same outcome happens twice in a row
    if len(history) >= 2 and history[-1] == history[-2]:
        return 1 - history[-1]  # Predict the opposite
    return history[-1]  # Otherwise, follow the most recent outcome

# Weighted Averaging (Combine the predictions of multiple strategies)
def weighted_averaging(predictions):
    return 1 if sum(predictions) > len(predictions) / 2 else 0

# Voting Classifier (Combines the above predictions)
def voting_classifier(history):
    predictions = [
        historical_pattern_recognition(history, len(history)),
        moving_average_prediction(history),
        support_resistance_prediction(history),
        trend_following(history),
        mean_reversion(history),
        time_series_analysis(history),
        momentum_prediction(history),
        rule_based_prediction(history)
    ]
    return weighted_averaging(predictions)

# Predict the next outcome based on the last N outcomes
def predict_next_outcome(period_number, history):
    return voting_classifier(history)

# Simulate predictions
def simulate_predictions():
    history = deque(maxlen=10)  # Use the last 10 outcomes as history
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(len(y)):
        if len(history) >= 5:  # Start predictions after having enough history
            predicted_outcome = predict_next_outcome(X[i], list(history))
            actual_outcome = y[i]
            if predicted_outcome == actual_outcome:
                correct_predictions += 1
            total_predictions += 1
        history.append(y[i])  # Update history with actual outcome
    
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Prediction accuracy: {accuracy:.2f}%")

# Run simulation
simulate_predictions()
