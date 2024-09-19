import streamlit as st
import random
import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

def calculate_probabilities(outcomes, window_size=None):
    if window_size:
        outcomes = outcomes[-window_size:]
    dragon_count = outcomes.count('Dragon')
    tiger_count = outcomes.count('Tiger')
    total = len(outcomes)
    
    dragon_prob = dragon_count / total if total > 0 else 0.5
    tiger_prob = tiger_count / total if total > 0 else 0.5
    
    return [dragon_prob, tiger_prob]

def detect_streaks(outcomes, streak_threshold=3):
    current_streak = 1
    last_outcome = outcomes[-1] if outcomes else None
    
    for outcome in reversed(outcomes[:-1]):
        if outcome == last_outcome:
            current_streak += 1
        else:
            break
    
    return current_streak >= streak_threshold, last_outcome

def prepare_historical_data(outcomes, one_hot_encoder):
    X, y = [], []
    for i in range(5, len(outcomes)):
        recent_trend = calculate_probabilities(outcomes[i-10:i])
        long_term_trend = calculate_probabilities(outcomes[:i])
        streak, last_outcome = detect_streaks(outcomes[:i])
        streak_feature = [1 if streak else 0, 1 if last_outcome == 'Dragon' else 0]
        pattern = one_hot_encoder.fit_transform([[o] for o in outcomes[i-5:i]])
        
        features = np.concatenate([recent_trend, long_term_trend, streak_feature, pattern.flatten()])
        X.append(features)
        y.append(1 if outcomes[i] == 'Dragon' else 0)
    
    return np.array(X), np.array(y)

def advanced_predict(outcomes, rf_model, one_hot_encoder):
    outcomes_list = list(outcomes)
    
    recent_trend = calculate_probabilities(outcomes_list, window_size=10)
    long_term_trend = calculate_probabilities(outcomes_list)
    streak, last_outcome = detect_streaks(outcomes_list)
    streak_feature = [1 if streak else 0, 1 if last_outcome == 'Dragon' else 0]
    pattern = one_hot_encoder.fit_transform([[o] for o in outcomes_list[-5:]])
    
    features = np.concatenate([recent_trend, long_term_trend, streak_feature, pattern.flatten()])
    
    X, y = prepare_historical_data(outcomes_list, one_hot_encoder)
    
    rf_model.fit(X, y)
    
    prediction = rf_model.predict_proba([features])[0]
    
    return prediction[1], prediction[0]  # Dragon probability, Tiger probability

def main():
    st.title("Advanced Dragon Tiger Prediction")

    if 'outcomes' not in st.session_state:
        st.session_state.outcomes = deque(maxlen=1000)
    
    if 'rf_model' not in st.session_state:
        st.session_state.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    if 'one_hot_encoder' not in st.session_state:
        st.session_state.one_hot_encoder = OneHotEncoder(sparse_output=False)

    st.subheader("Input Multiple Outcomes")
    input_text = st.text_input("Enter outcomes (D for Dragon, T for Tiger, separated by spaces):")
    if st.button("Add Multiple Outcomes"):
        new_outcomes = input_text.upper().split()
        for outcome in new_outcomes:
            if outcome == 'D':
                st.session_state.outcomes.append("Dragon")
            elif outcome == 'T':
                st.session_state.outcomes.append("Tiger")
        st.success(f"Added {len(new_outcomes)} outcomes")

    st.subheader("Quick Input for Last Few Outcomes")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("D🐉"):
            st.session_state.outcomes.append("Dragon")
    with col2:
        if st.button("T🐅"):
            st.session_state.outcomes.append("Tiger")
    with col3:
        if st.button("DD🐉🐉"):
            st.session_state.outcomes.extend(["Dragon", "Dragon"])
    with col4:
        if st.button("TT🐅🐅"):
            st.session_state.outcomes.extend(["Tiger", "Tiger"])
    with col5:
        if st.button("DT🐉🐅"):
            st.session_state.outcomes.extend(["Dragon", "Tiger"])

    st.subheader("Current Outcomes")
    st.write(f"Total outcomes recorded: {len(st.session_state.outcomes)}")
    st.write("Last 10 outcomes:")
    for outcome in list(st.session_state.outcomes)[-10:]:
        st.write(f"{'🐉' if outcome == 'Dragon' else '🐅'}")

    if len(st.session_state.outcomes) >= 10:
        if st.button("Predict Next Outcome"):
            dragon_prob, tiger_prob = advanced_predict(
                st.session_state.outcomes, 
                st.session_state.rf_model, 
                st.session_state.one_hot_encoder
            )
            prediction = "Dragon 🐉" if dragon_prob > tiger_prob else "Tiger 🐅"
            st.success(f"Predicted next outcome: {prediction}")
            st.write(f"Probabilities: Dragon {dragon_prob:.2f}, Tiger {tiger_prob:.2f}")
    else:
        st.warning(f"Need at least 10 outcomes for advanced prediction. Current count: {len(st.session_state.outcomes)}")

    if st.button("Reset All Outcomes"):
        st.session_state.outcomes.clear()
        st.experimental_rerun()

if __name__ == "__main__":
    main()
