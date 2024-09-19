import streamlit as st
import random
from collections import defaultdict
import json
import os
from datetime import datetime, timedelta

# Constants
HISTORY_FILE = "dragon_tiger_history.json"
MAX_HISTORY = 100
ROUND_DURATION = 30  # seconds

def load_history():
    """Load game history from a JSON file."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = []
    return history[-MAX_HISTORY:]

def save_history(history):
    """Save game history to a JSON file."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def predict_next_outcome(last_outcomes):
    """Predict the next outcome based on the last outcomes."""
    # Placeholder for prediction logic; implement your own logic here.
    # Example: return random.choice(['d', 't'])  # Random prediction for demonstration
    pass  # Keep the existing prediction logic as per your original code

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Dragon vs Tiger Predictor", page_icon="🐉", layout="wide")
    st.title("Real-time Dragon vs Tiger Predictor")

    # Initialize session state variables
    if 'history' not in st.session_state:
        st.session_state.history = load_history()
    if 'next_round_time' not in st.session_state:
        st.session_state.next_round_time = datetime.now() + timedelta(seconds=ROUND_DURATION)
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None

    # Instructions for users
    st.markdown("### How to use:")
    st.markdown("1. Wait for the timer to count down to 0.")
    st.markdown("2. When a new round starts, quickly click 'Dragon' or 'Tiger' based on the outcome.")
    st.markdown("3. The app will immediately show a prediction for the next round.")

    # Create buttons for user interaction
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Dragon 🐉", key="dragon_button"):
            st.session_state.history.append('d')
            save_history(st.session_state.history)
            st.session_state.next_round_time = datetime.now() + timedelta(seconds=ROUND_DURATION)

    with col2:
        if st.button("Tiger 🐅", key="tiger_button"):
            st.session_state.history.append('t')
            save_history(st.session_state.history)
            st.session_state.next_round_time = datetime.now() + timedelta(seconds=ROUND_DURATION)

    with col3:
        if st.button("Reset Timer", key="reset_button"):
            st.session_state.next_round_time = datetime.now() + timedelta(seconds=ROUND_DURATION)

    # Display recent history
    st.write("Recent History:")
    recent_outcomes = ['🐉' if outcome == 'd' else '🐅' for outcome in st.session_state.history[-10:]
                       ]
    st.write(" ".join(recent_outcomes))

    # Make predictions based on history
    if st.session_state.history:
        st.session_state.last_prediction = predict_next_outcome(st.session_state.history)
        prediction = "Dragon 🐉" if st.session_state.last_prediction == 'd' else "Tiger 🐅"
        confidence = abs(0.5 - (st.session_state.history.count('d') / len(st.session_state.history)))
        st.subheader(f"Prediction for next round: {prediction}")
        st.write(f"Confidence: {confidence * 100:.2f}%")

    # Display time until the next round
    time_left = max(0, (st.session_state.next_round_time - datetime.now()).total_seconds())
    if time_left > 0:
        st.write(f"Time until next round: {int(time_left)} seconds")
    else:
        st.write("New round started! Please enter the result quickly.")

    # Force a rerun every second to update the timer
    st.empty()
    st.experimental_rerun()

if __name__ == "__main__":
    main()
