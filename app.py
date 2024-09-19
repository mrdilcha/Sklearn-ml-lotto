import streamlit as st
import random
from collections import deque

def calculate_probabilities(outcomes, window_size=10):
    recent_outcomes = outcomes[-window_size:]
    dragon_count = recent_outcomes.count('Dragon')
    tiger_count = recent_outcomes.count('Tiger')
    total = len(recent_outcomes)
    
    dragon_prob = dragon_count / total if total > 0 else 0.5
    tiger_prob = tiger_count / total if total > 0 else 0.5
    
    return dragon_prob, tiger_prob

def detect_streaks(outcomes, streak_threshold=3):
    current_streak = 1
    last_outcome = outcomes[0] if outcomes else None
    
    for outcome in outcomes[1:]:
        if outcome == last_outcome:
            current_streak += 1
        else:
            current_streak = 1
        last_outcome = outcome
        
    return current_streak >= streak_threshold, last_outcome

def predict_next_outcome(outcomes):
    if len(outcomes) < 5:
        return random.choice(["Dragon 🐉", "Tiger 🐅"])
    
    dragon_prob, tiger_prob = calculate_probabilities(outcomes)
    streak_detected, streak_outcome = detect_streaks(outcomes)
    
    if streak_detected:
        # If there's a streak, slightly increase the probability of it breaking
        if streak_outcome == 'Dragon':
            tiger_prob += 0.1
        else:
            dragon_prob += 0.1
    
    # Consider the overall trend
    if len(outcomes) >= 20:
        long_term_dragon_prob, long_term_tiger_prob = calculate_probabilities(outcomes, window_size=20)
        dragon_prob = 0.7 * dragon_prob + 0.3 * long_term_dragon_prob
        tiger_prob = 0.7 * tiger_prob + 0.3 * long_term_tiger_prob
    
    # Normalize probabilities
    total_prob = dragon_prob + tiger_prob
    dragon_prob /= total_prob
    tiger_prob /= total_prob
    
    # Make the prediction
    if random.random() < dragon_prob:
        return "Dragon 🐉"
    else:
        return "Tiger 🐅"

def main():
    st.title("Dragon Tiger Prediction")

    if 'outcomes' not in st.session_state:
        st.session_state.outcomes = deque(maxlen=100)  # Store up to 100 recent outcomes

    if 'reset' not in st.session_state:
        st.session_state.reset = False

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Dragon 🐉"):
            st.session_state.outcomes.append("Dragon")
            st.session_state.reset = False

    with col2:
        if st.button("Tiger 🐅"):
            st.session_state.outcomes.append("Tiger")
            st.session_state.reset = False

    st.write("Last 5 outcomes:")
    for outcome in list(st.session_state.outcomes)[-5:]:
        st.write(f"{'🐉' if outcome == 'Dragon' else '🐅'}")

    if len(st.session_state.outcomes) >= 5:
        if st.button("Predict Next Outcome"):
            prediction = predict_next_outcome(list(st.session_state.outcomes))
            st.success(f"Predicted next outcome: {prediction}")

            # Display probabilities
            dragon_prob, tiger_prob = calculate_probabilities(list(st.session_state.outcomes))
            st.write(f"Current probabilities: Dragon {dragon_prob:.2f}, Tiger {tiger_prob:.2f}")

    if st.button("Reset"):
        st.session_state.reset = True

    if st.session_state.reset:
        st.session_state.outcomes.clear()
        st.session_state.reset = False
        st.rerun()

if __name__ == "__main__":
    main()
