import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import streamlit as st

# Load dataset (assumed structure: 'period_number', 'outcome')
# Outcome: 1 for Big, 0 for Small
@st.cache
def load_data():
    data = pd.read_csv('data.csv')
    return data

data = load_data()

# Feature and label extraction
X = data['period_number'].values.reshape(-1, 1)  # We can add more features if available
y = data['outcome'].values  # 0: Small, 1: Big

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
random_forest = RandomForestClassifier(n_estimators=100)
logistic_regression = LogisticRegression()
svc = SVC(probability=True)
knn = KNeighborsClassifier(n_neighbors=5)
naive_bayes = GaussianNB()
gradient_boosting = GradientBoostingClassifier()

# Create an ensemble VotingClassifier with hard voting
voting_clf = VotingClassifier(estimators=[
    ('rf', random_forest),
    ('lr', logistic_regression),
    ('svc', svc),
    ('knn', knn),
    ('gnb', naive_bayes),
    ('gb', gradient_boosting)
], voting='hard')

# Train the classifiers
voting_clf.fit(X_train, y_train)

# Streamlit UI
st.title('Big/Small Outcome Predictor')

st.write(f"Prediction accuracy: {accuracy_score(y_test, voting_clf.predict(X_test)) * 100:.2f}%")

period_number_input = st.text_input("Enter the next period number (last three digits):")

# Function to predict the outcome based on the period number
def predict_next_outcome(period_number):
    prediction = voting_clf.predict([[period_number]])
    return "Big" if prediction == 1 else "Small"

if st.button('Predict'):
    if period_number_input.isdigit():
        next_period_number = int(period_number_input)
        predicted_outcome = predict_next_outcome(next_period_number)
        st.write(f"The predicted outcome for period {next_period_number} is: {predicted_outcome}")
    else:
        st.write("Please enter a valid number")
