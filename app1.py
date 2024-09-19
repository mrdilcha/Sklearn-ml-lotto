import streamlit as st
import random
import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# ... (keep the helper functions: calculate_probabilities, detect_streaks, prepare_historical_data, advanced_predict)

def main():
    st.title("Advanced Dragon Tiger Prediction")

    if 'outcomes' not in st.session_state:
        st.session_state.outcomes = deque(maxlen=1000)
    
    if 'rf_model' not in st.session_state:
        st.session_state.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    if 'one_hot_encoder' not in st.session_state:
        st.session_state.one_hot_encoder = OneHotEncoder(sparse=False)

    # New feature: Input multiple outcomes at once
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

    # New feature: Quick input for last few outcomes
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
