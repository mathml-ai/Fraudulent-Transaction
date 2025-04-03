import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")

# Initialize label encoder
le = LabelEncoder()

def preprocess_input(data):
    """Preprocess input dictionary to match model features."""
    df = pd.DataFrame([data])
    
    # Apply transformations
    df['newbalanceOrig_transformed'] = np.log1p(df['newbalanceOrig'])
    df['newbalanceOrig_flag'] = (df['newbalanceOrig_transformed'] < 5).astype(int)
    
    df['oldbalanceOrg_transformed'] = np.log1p(df['oldbalanceOrg'])
    
    df['amount_transformed'] = np.log1p(df['amount'])
    
    df['newbalanceDest_transformed'] = np.log1p(df['newbalanceDest'])
    
    df['type'] = le.fit_transform([df['type'][0]])[0]  # Ensure correct transformation
    df['type_1_flag'] = (df['type'] == 1).astype(int)
    
    # Select relevant columns
    df = df[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
             'newbalanceDest', 'newbalanceOrig_transformed', 'newbalanceOrig_flag',
             'oldbalanceOrg_transformed', 'amount_transformed', 'newbalanceDest_transformed',
             'type_1_flag']]
    
    return df

# Streamlit UI
st.title("Fraud Detection System")
st.write("Enter transaction details to predict if it's fraudulent.")

# Input fields
step = st.number_input("Step (Time Step)", min_value=0, value=1)
type_value = st.selectbox("Transaction Type", ['CASH_OUT', 'PAYMENT', 'TRANSFER', 'DEBIT', 'CASH_IN'])
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
oldbalanceOrg = st.number_input("Old Balance of Origin Account", min_value=0.0, value=1000.0)
newbalanceOrig = st.number_input("New Balance of Origin Account", min_value=0.0, value=900.0)
newbalanceDest = st.number_input("New Balance of Destination Account", min_value=0.0, value=2000.0)

# Prepare input data
input_data = {
    "step": step,
    "type": type_value,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "newbalanceDest": newbalanceDest
}

if st.button("Predict Fraud"):
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    result = "Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction"
    st.write(f"Prediction: {result}")
