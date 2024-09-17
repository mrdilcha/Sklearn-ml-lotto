import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Generate synthetic historical data for training
def generate_data(num_samples=1000):
    np.random.seed(42)
    data = []
    
    for _ in range(num_samples):
        last_three_digits = np.random.randint(100, 1000)
        digit_sum = sum(int(digit) for digit in str(last_three_digits))
        outcome = "big" if np.random.rand() > 0.5 else "small"
        data.append([last_three_digits, digit_sum, outcome])
    
    return pd.DataFrame(data, columns=["last_three_digits", "digit_sum", "outcome"])

# Prepare the dataset
df = generate_data()
df['outcome'] = df['outcome'].map({"big": 1, "small": 0})

X = df[['last_three_digits', 'digit_sum']]
y = df['outcome']

# Train the logistic regression model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression()
model.fit(X_scaled, y)

def predict_outcome(last_three_digits):
    digit_sum = sum(int(digit) for digit in str(last_three_digits))
    features = np.array([[last_three_digits, digit_sum]])
    scaled_features = scaler.transform(features)
    
    prediction = model.predict(scaled_features)
    return "big" if prediction[0] == 1 else "small"

# Streamlit app layout
st.title("Big or Small Prediction Game")
st.write("Enter the last three digits of the period number to predict the outcome.")

last_three_digits = st.text_input("Last three digits:")

if st.button("Predict"):
    if len(last_three_digits) == 3 and last_three_digits.isdigit():
        prediction = predict_outcome(int(last_three_digits))
        actual_outcome = np.random.choice(["big", "small"])
        
        st.write(f"Your prediction: **{prediction}**")
        st.write(f"Actual outcome: **{actual_outcome}**")
        
        if prediction == actual_outcome:
            st.success("Congratulations! You predicted correctly.")
        else:
            st.error("Sorry, better luck next time.")
    else:
        st.warning("Please enter exactly three digits.")
        
