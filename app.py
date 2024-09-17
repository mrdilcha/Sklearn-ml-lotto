import streamlit as st
import pandas as pd
import numpy as np
from collections import deque

# Add a title to your app
st.title("Prediction Application")

# Function definitions from the first snippet
def moving_average(arr, window_size=5):
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

def historical_pattern_recognition(history, period):
    pattern_length = 5
    if len(history) < pattern_length:
        return np.random.choice([0, 1])
    pattern = history[-pattern_length:]
    for i in range(len(history) - pattern_length):
        if np.array_equal(history[i:i+pattern_length], pattern):
            return history[i+pattern_length]
    return np.random.choice([0, 1])

def moving_average_prediction(history):
    ma = moving_average(history, window_size=3)
    if len(ma) == 0:
        return np.random.choice([0, 1])
    return 1 if ma[-1] > 0.5 else 0

def support_resistance_prediction(history):
    small_count = np.sum(np.array(history) == 0)
    big_count = np.sum(np.array(history) == 1)
    return 0 if small_count > big_count else 1

def trend_following(history):
    if len(history) < 3:
        return np.random.choice([0, 1])
    return history[-1]

def mean_reversion(history):
    consecutive = 3
    if len(history) >= consecutive and all(x == history[-1] for x in history[-consecutive:]):
        return 1 - history[-1]
    return np.random.choice([0, 1])

def time_series_analysis(history):
    lag = 2
    if len(history) < lag:
        return np.random.choice([0, 1])
    return history[-lag]

def momentum_prediction(history):
    if len(history) < 3:
        return np.random.choice([0, 1])
    momentum = sum(history[-3:])
    return 1 if momentum > 1 else 0

def rule_based_prediction(history):
    if len(history) == 0:
        return np.random.choice([0, 1])
    if len(history) >= 2 and history[-1] == history[-2]:
        return 1 - history[-1]
    return history[-1]

def weighted_averaging(predictions):
    return 1 if sum(predictions) > len(predictions) / 2 else 0

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

def predict_next_outcome(period_number, history):
    return voting_classifier(history)

# Load dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write("Here are the first few rows of your data:")
    st.write(data.head())
    
    # Feature and label extraction
    X = data['period_number'].values
    y = data['outcome'].values  # 0: Small, 1: Big
    
    # Add a button to run the prediction
    if st.button("Run Prediction"):
        st.write("Running Prediction...")
        
        correct_predictions = 0
        total_predictions = 0
        history = deque(maxlen=10)
        
        for i in range(len(y)):
            if len(history) >= 5:
                predicted_outcome = predict_next_outcome(X[i], list(history))
                actual_outcome = y[i]
                if predicted_outcome == actual_outcome:
                    correct_predictions += 1
                total_predictions += 1
            history.append(y[i])
        
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        st.write(f"Prediction Accuracy: {accuracy:.2f}%")
else:
    st.write("Please upload a CSV file to continue.")
