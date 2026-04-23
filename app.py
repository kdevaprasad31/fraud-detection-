import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv('creditcard(4).csv')

# -------------------------------
# FEATURES & TARGET
# -------------------------------
X = df.drop('Class', axis=1)
y = df['Class']

# -------------------------------
# SCALING
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# MODEL TRAINING
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("💳 Credit Card Fraud Detection App")

st.write("Enter transaction details below:")

# Take user input dynamically
input_data = []

for col in X.columns:
    value = st.number_input(f"{col}", value=0.0)
    input_data.append(value)

# Convert input to array
input_array = np.array(input_data).reshape(1, -1)

# Scale input
input_scaled = scaler.transform(input_array)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")