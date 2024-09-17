import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import streamlit as st

# Function to generate balanced synthetic historical data for training
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

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize classifiers
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(probability=True)
nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

# Train models and evaluate accuracy
rf_model.fit(X_train_scaled, y_train)
svm_model.fit(X_train_scaled, y_train)
nn_model.fit(X_train_scaled, y_train)

# Function to make predictions based on user input
def predict_outcome(last_three_digits):
    digit_sum = sum(int(digit) for digit in str(last_three_digits))
    features = np.array([[last_three_digits, digit_sum]])
    scaled_features = scaler.transform(features)
    
    rf_prediction = rf_model.predict(scaled_features)[0]
    svm_prediction = svm_model.predict(scaled_features)[0]
    nn_prediction = nn_model.predict(scaled_features)[0]
    
    # Check if any two models agree on the prediction
    if (rf_prediction == svm_prediction) or (rf_prediction == nn_prediction) or (svm_prediction == nn_prediction):
        return "big" if rf_prediction == svm_prediction == nn_prediction == 1 else "small"
    else:
        return "Inconclusive"

# Streamlit app layout
st.title("Outcome Prediction using Multiple Algorithms")
st.write("Enter the last three digits of the period number to predict the outcome.")

last_three_digits_input = st.text_input("Last three digits:", "")

if st.button("Predict"):
    if len(last_three_digits_input) == 3 and last_three_digits_input.isdigit():
        prediction = predict_outcome(int(last_three_digits_input))
        st.write(f"Prediction: **{prediction}**")
    else:
        st.warning("Please enter exactly three digits.")
