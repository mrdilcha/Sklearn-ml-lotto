import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import streamlit as st

# Function to generate synthetic data
def generate_data(num_samples=10000):
    np.random.seed(42)
    data = []
    
    for _ in range(num_samples):
        last_three_digits = np.random.randint(100, 1000)
        outcome = 1 if np.random.rand() > 0.5 else 0  # Randomly assign big (1) or small (0)
        data.append([last_three_digits, outcome])
    
    return pd.DataFrame(data, columns=["last_three_digits", "outcome"])

# Prepare the dataset
df = generate_data(10000)
X = df[['last_three_digits']]
y = df['outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(1,)))  # Input layer
model.add(Dense(10, activation='relu'))                     # Hidden layer
model.add(Dense(1, activation='sigmoid'))                   # Output layer

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32)

# Function to predict outcome based on user input
def predict_outcome(last_three_digits):
    scaled_input = scaler.transform(np.array([[last_three_digits]]))  # Scale input
    prediction = model.predict(scaled_input)
    return "big" if prediction[0][0] > 0.5 else "small"

# Streamlit app layout
st.title("Big or Small Prediction App")
st.write("Enter the last three digits of the period number to predict the outcome.")

# Input field for user to enter last three digits
last_three_digits_input = st.text_input("Last three digits:", "")

if st.button("Predict"):
    if last_three_digits_input.isdigit() and len(last_three_digits_input) == 3:
        prediction_result = predict_outcome(int(last_three_digits_input))
        st.write(f"Prediction: **{prediction_result}**")
    else:
        st.warning("Please enter exactly three digits.")
