import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import numpy as np
import smtplib
from email.mime.text import MIMEText
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fraud Detector", layout="centered")

# Load API and Email config
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()
DATA_PATH = "Banking Transactions Data For Fraud.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)

data = load_data()

# Drop identifier columns
columns_to_drop = ["transaction_id", "device_id"]
data = data.drop(columns=columns_to_drop)

# Encode categorical variables
categorical_cols = data.select_dtypes(include=['object', 'bool']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

X = data.copy()

# Train Isolation Forest
isolation_model = IsolationForest(contamination=0.05, random_state=42)
isolation_model.fit(X)

# Prepare scored data
X_scored = X.copy()
X_scored["anomaly_score"] = isolation_model.decision_function(X)
X_scored["is_fraud"] = isolation_model.predict(X)

# Behavioral Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
X_scored["behavior_cluster"] = kmeans.fit_predict(X)

def sanitize_numeric(value):
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").strip()
    try:
        return float(value)
    except:
        return 0.0

def standardize_categoricals(user_input):
    if "is_international" in user_input:
        val = user_input["is_international"].strip().lower()
        user_input["is_international"] = "Yes" if val in ["yes", "y", "true", "1"] else "No"
    return user_input

def send_email_alert(to_email, subject, message):
    sender_email = os.getenv("ALERT_SENDER_EMAIL")
    sender_password = os.getenv("ALERT_SENDER_PASSWORD")
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
            st.success("Email alert sent!")
    except Exception as e:
        st.error(f"Email alert failed: {e}")

def predict_fraud(user_input):
    user_input = standardize_categoricals(user_input)

    if "account_age_days" in user_input:
        user_input["account_age_days"] = sanitize_numeric(user_input["account_age_days"])
    if "transaction_duration" in user_input:
        user_input["transaction_duration"] = sanitize_numeric(user_input["transaction_duration"])

    for key in ["transaction_amount", "balance_before_transaction", "balance_after_transaction", "customer_age", "login_attempts"]:
        if key in user_input:
            user_input[key] = sanitize_numeric(user_input[key])

    full_row = {col: user_input.get(col, 0 if col not in categorical_cols else "Unknown") for col in X.columns}
    input_df = pd.DataFrame([full_row])

    for col in categorical_cols:
        if col in input_df:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
            except ValueError:
                fallback = label_encoders[col].classes_[0]
                input_df[col] = [label_encoders[col].transform([fallback])[0]]

    input_df = input_df.astype(float)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    prediction = isolation_model.predict(input_df)[0]
    fraud_score = round(abs(isolation_model.decision_function(input_df)[0]), 2)
    behavior_cluster = kmeans.predict(input_df)[0]
    result = 1 if prediction == -1 else 0
    return result, fraud_score, behavior_cluster

# UI Setup
st.markdown("## Fraud Detection Chatbot")

with st.form("user_input_form"):
    st.markdown("### Enter transaction data below:")
    user_input = {}
    for col in X.columns:
        if col in categorical_cols:
            user_input[col] = st.text_input(f"{col}:")
        else:
            user_input[col] = st.text_input(f"{col} (numeric):")

    submitted = st.form_submit_button("Analyze Transaction")

if submitted:
    prediction, fraud_score, behavior_cluster = predict_fraud(user_input)
    result = "Fraudulent" if prediction == 1 else "Not Fraudulent"

    if prediction == 1:
        send_email_alert(
            to_email="your_recipient@example.com",
            subject="Fraud Alert",
            message=f"Potential fraud detected:\n\n{user_input}\n\nFraud Score: {fraud_score}"
        )

    st.markdown(f"### Prediction: {result}")
    st.markdown(f"**Fraud Score:** {fraud_score}")
    st.markdown(f"**Behavioral Cluster:** {behavior_cluster}")

    prompt = (
        f"Given the transaction data: {user_input},\n"
        f"and a model that flagged it as {result} with fraud score {fraud_score},\n"
        "explain which behavioral and financial patterns may have contributed to this classification."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful fraud risk advisor who explains AI-based anomaly detection decisions."},
            {"role": "user", "content": prompt}
        ]
    )
    explanation = response.choices[0].message.content

    st.markdown("---")
    st.markdown("### Risk Assessment Explanation:")
    st.markdown(explanation)

    top_features = X_scored.drop(columns=["anomaly_score", "is_fraud", "behavior_cluster"]).corrwith(X_scored["anomaly_score"]).abs().sort_values(ascending=False).head(10).index
    heatmap_data = X_scored[top_features].copy()
    heatmap_data["anomaly_score"] = X_scored["anomaly_score"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Anomaly Score Heatmap (Top Correlated Features)")
    st.pyplot(fig)
