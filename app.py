
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("WATSON_API_KEY")
DEPLOYMENT_URL = os.getenv("WATSON_DEPLOYMENT_URL")

def call_ibm_granite(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 300
        }
    }
    try:
        response = requests.post(DEPLOYMENT_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("results", [{}])[0].get("generated_text", "No response.")
    except Exception as e:
        return f"Error: {str(e)}"

def answer_patient_query(query):
    prompt = f"You are a medical assistant. Answer this question clearly and professionally:\n\nQuestion: {query}"
    return call_ibm_granite(prompt)

def predict_disease(symptoms):
    prompt = f"A patient is reporting the following symptoms: {symptoms}. What are the most likely diseases? Provide top 3 with probability and next steps."
    return call_ibm_granite(prompt)

def generate_treatment_plan(condition, age, gender):
    prompt = f"Generate a treatment plan for a {age}-year-old {gender} diagnosed with {condition}. Include medications, lifestyle advice, and follow-up care."
    return call_ibm_granite(prompt)

def display_health_analytics():
    st.subheader("Health Metrics Dashboard")
    df = pd.DataFrame({
        "Date": pd.date_range(start="2025-07-01", periods=7),
        "Heart Rate": [72, 74, 76, 75, 78, 80, 77],
        "Blood Pressure": [120, 122, 118, 125, 128, 124, 122],
        "Blood Glucose": [95, 100, 102, 99, 105, 110, 108]
    })

    fig1 = px.line(df, x="Date", y="Heart Rate", title="Heart Rate Over Time")
    fig2 = px.line(df, x="Date", y="Blood Pressure", title="Blood Pressure Over Time")
    fig3 = px.line(df, x="Date", y="Blood Glucose", title="Blood Glucose Over Time")

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)

# ------------------------- UI -------------------------

st.set_page_config(page_title="HealthAI", layout="wide")
st.title("🩺 HealthAI - Intelligent Healthcare Assistant")

menu = st.sidebar.radio("Navigate", ["Patient Chat", "Disease Prediction", "Treatment Plans", "Health Analytics"])

if menu == "Patient Chat":
    st.subheader("💬 Ask a Health Question")
    query = st.text_area("Enter your question:")
    if st.button("Get Answer"):
        if query:
            response = answer_patient_query(query)
            st.success(response)

elif menu == "Disease Prediction":
    st.subheader("🧠 Disease Prediction")
    symptoms = st.text_area("Enter your symptoms (comma separated):")
    if st.button("Predict Disease"):
        if symptoms:
            result = predict_disease(symptoms)
            st.info(result)

elif menu == "Treatment Plans":
    st.subheader("📝 Treatment Plan Generator")
    condition = st.text_input("Enter your condition (e.g., Diabetes):")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    if st.button("Generate Plan"):
        if condition:
            plan = generate_treatment_plan(condition, age, gender)
            st.success(plan)

elif menu == "Health Analytics":
    display_health_analytics()
