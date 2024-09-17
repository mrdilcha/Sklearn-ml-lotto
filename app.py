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
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(probability=True),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
}

# Train models and evaluate accuracy
accuracies = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    accuracies[name] = accuracy_score(y_test, predictions)

# Display accuracies in Streamlit app
st.title("Outcome Prediction using Multiple Algorithms")
st.write("Accuracy of different models on test data:")
for model_name, accuracy in accuracies.items():
    st.write(f"{model_name}: {accuracy:.2f}")

# Function to make predictions based on user input
def predict_outcome(model_name, last_three_digits):
    digit_sum = sum(int(digit) for digit in str(last_three_digits))
    features = np.array([[last_three_digits, digit_sum]])
    scaled_features = scaler.transform(features)
    
    model = models[model_name]
    prediction = model.predict(scaled_features)[0]
    
    return "big" if prediction == 1 else "small"

# Streamlit app layout for user prediction input
st.write("Enter the last three digits of the period number to predict the outcome.")
last_three_digits_input = st.text_input("Last three digits:", "")

selected_model = st.selectbox("Select Model:", list(models.keys()))

if st.button("Predict"):
    if len(last_three_digits_input) == 3 and last_three_digits_input.isdigit():
        prediction = predict_outcome(selected_model, int(last_three_digits_input))
        st.write(f"Your prediction using {selected_model}: **{prediction}**")
    else:
        st.warning("Please enter exactly three digits.")
