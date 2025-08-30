import streamlit as st
import pandas as pd
import pickle
import numpy as np

with open("xgb_attrition_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("model_columns.pkl", "rb") as file:
    model_columns = pickle.load(file)

st.title("Employee Attrition Prediction")
st.write("Predict whether an employee is likely to leave the company.")


def user_input_features():
    inputs = {}
    for col in model_columns:
       
        if "Age" in col or "Rate" in col or "Years" in col:
            inputs[col] = st.number_input(f"{col}", min_value=0, max_value=100, value=30)
        else:
            
            inputs[col] = st.selectbox(f"{col}", [0, 1])
    return pd.DataFrame([inputs])

input_df = user_input_features()

if st.button("Predict Attrition"):
    
    proba = model.predict_proba(input_df)[:, 1][0]
    
    threshold = 0.3
    prediction = int(proba >= threshold)
    
    st.write(f"**Probability of Attrition:** {proba:.2f}")
    st.write(f"**Prediction (Attrition = 1, Stay = 0):** {prediction}")
    
    if prediction == 1:
        st.warning("⚠️ This employee is likely to leave.")
    else:
        st.success("✅ This employee is likely to stay.")
