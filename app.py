import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Generate balanced synthetic historical data for training
def generate_balanced_data(num_samples=1000):
    np.random.seed(42)
    data = []
    
    # Create an equal number of big and small outcomes
    for _ in range(num_samples // 2):
        last_three_digits = np.random.randint(100, 1000)
        digit_sum = sum(int(digit) for digit in str(last_three_digits))
        data.append([last_three_digits, digit_sum, "big"])
        
        last_three_digits = np.random.randint(100, 1000)
        digit_sum = sum(int(digit) for digit in str(last_three_digits))
        data.append([last_three_digits, digit_sum, "small"])
    
    return pd.DataFrame(data, columns=["last_three_digits", "digit_sum", "outcome"])

# Prepare the dataset
df = generate_balanced_data()
df['outcome'] = df['outcome'].map({"big": 1, "small": 0})

X = df[['last_three_digits', 'digit_sum']]
y = df['outcome']

# Train the Random Forest classifier
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

def predict_outcome(last_three_digits):
    digit_sum = sum(int(digit) for digit in str(last_three_digits))
    features = np.array([[last_three_digits, digit_sum]])
    scaled_features = scaler.transform(features)
    
    prediction = model.predict(scaled_features)
    
    # Introduce randomness to predictions
    if np.random.rand() < 0.1:  # 10% chance to flip prediction
        return "small" if prediction[0] == 1 else "big"
    
    return "big" if prediction[0] == 1 else "small"

# Streamlit app layout
st.title("Big or Small Prediction Game")
st.write("Enter the last three digits of the period number to predict the outcome.")

last_three_digits = st.text_input("Last three digits:")

if st.button("Predict"):
    if len(last_three_digits) == 3 and last_three_digits.isdigit():
        prediction = predict_outcome(int(last_three_digits))
        st.write(f"Your prediction: **{prediction}**")
    else:
        st.warning("Please enter exactly three digits.")
