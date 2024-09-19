import streamlit as st
import random
from collections import defaultdict
import json
import os
from datetime import datetime, timedelta

HISTORY_FILE = "dragon_tiger_history.json"
MAX_HISTORY = 100

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = []
    return history[-MAX_HISTORY:]  # Keep only the last MAX_HISTORY outcomes

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def predict_next_outcome(last_outcomes):
    if len(last_outcomes) < 2:
        return random.choice(['d', 't'])

    transitions = defaultdict(lambda: {'d': 0, 't': 0})
    for i in range(len(last_outcomes) - 1):
        current, next_outcome = last_outcomes[i:i+2]
        transitions[current][next_outcome] += 1

    for state in transitions:
        total = sum(transitions[state].values())
        for outcome in transitions[state]:
            transitions[state][outcome] /= total

    recent_history = last_outcomes[-10:]
    weights = [1.1 ** i for i in range(len(recent_history))]
    weighted_counts = defaultdict(float)
    for outcome, weight in zip(recent_history, reversed(weights)):
        weighted_counts[outcome] += weight

    last_outcome = last_outcomes[-1]
    markov_prob_dragon = transitions[last_outcome].get('d', 0.5)
    recent_prob_dragon = weighted_counts['d'] / sum(weighted_counts.values())

    combined_prob_dragon = 0.7 * markov_prob_dragon + 0.3 * recent_prob_dragon

    pattern = ''.join(last_outcomes[-3:])
    if len(last_outcomes) >= 6 and pattern == ''.join(last_outcomes[-6:-3]):
        return 'd' if pattern.count('t') > pattern.count('d') else 't'

    return 'd' if combined_prob_dragon > 0.5 else 't'

def main():
    st.set_page_config(page_title="Dragon vs Tiger Predictor", page_icon="🐉")
    st.title("Real-time Dragon vs Tiger Predictor")

    if 'history' not in st.session_state:
        st.session_state.history = load_history()

    if 'next_round_time' not in st.session_state:
        st.session_state.next_round_time = datetime.now() + timedelta(seconds=30)

    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Dragon (D)"):
            st.session_state.history.append('d')
            save_history(st.session_state.history)
            st.session_state.next_round_time = datetime.now() + timedelta(seconds=30)

    with col2:
        if st.button("Tiger (T)"):
            st.session_state.history.append('t')
            save_history(st.session_state.history)
            st.session_state.next_round_time = datetime.now() + timedelta(seconds=30)

    st.write("Recent History:")
    st.write(" ".join(['🐉' if outcome == 'd' else '🐅' for outcome in st.session_state.history[-10:]]))

    if st.session_state.history:
        st.session_state.last_prediction = predict_next_outcome(st.session_state.history)
        prediction = "Dragon 🐉" if st.session_state.last_prediction == 'd' else "Tiger 🐅"
        confidence = abs(0.5 - (st.session_state.history.count('d') / len(st.session_state.history)))
        st.subheader(f"Prediction for next round: {prediction}")
        st.write(f"Confidence: {confidence * 100:.2f}%")

    time_left = (st.session_state.next_round_time - datetime.now()).total_seconds()
    if time_left > 0:
        st.write(f"Time until next round: {int(time_left)} seconds")
    else:
        st.write("Time's up! Please enter the result for the new round.")

if __name__ == "__main__":
    main()
