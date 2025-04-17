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
import re

st.set_page_config(page_title="Fraud Detector", layout="centered")
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

DATA_PATH = "Banking Transactions Data For Fraud.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)

data = load_data()
data = data.drop(columns=["transaction_id", "branch_code", "device_id"])

data["is_negative_balance_after"] = (data["balance_after_transaction"] < 0).astype(int)
data["is_late_night"] = data["time_of_day"].apply(lambda x: 1 if str(x).lower() in ["night", "evening"] else 0)

categorical_cols = data.select_dtypes(include=['object', 'bool']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

features = [
    "transaction_amount", "is_international", "transaction_method", "time_of_day",
    "account_age_days", "login_attempts", "balance_before_transaction",
    "balance_after_transaction", "transaction_duration", "customer_age",
    "is_negative_balance_after", "is_late_night"
]
X = data[features]

model = IsolationForest(contamination=0.05, n_estimators=200, max_features=0.9, random_state=42)
model.fit(X)

X_scored = X.copy()
X_scored["anomaly_score"] = model.decision_function(X)
X_scored["is_fraud"] = model.predict(X)

kmeans = KMeans(n_clusters=4, random_state=42)
X_scored["behavior_cluster"] = kmeans.fit_predict(X)

def parse_numeric(text):
    try:
        return float(re.search(r"\d*\.?\d+", str(text)).group())
    except:
        return 0.0

def parse_account_age(text):
    text = str(text).lower()
    num = parse_numeric(text)
    return num * 30 if "month" in text else num * 365 if "year" in text else num

def parse_transaction_duration(text):
    text = str(text).lower()
    num = parse_numeric(text)
    return num / 60 if "second" in text else num * 60 if "hour" in text else num

def parse_yes_no(value):
    return 1 if str(value).strip().lower() in ['yes', 'y', 'true', '1'] else 0

def send_email_alert(to_email, subject, message):
    try:
        sender_email = os.getenv("ALERT_SENDER_EMAIL")
        sender_password = os.getenv("EMAIL_PASS")
        admin_email = os.getenv("ALERT_ADMIN_EMAIL")

        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to_email

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [to_email, admin_email], msg.as_string())
        return True
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False

if "result_data" not in st.session_state:
    st.session_state.result_data = {}
if "email_sent" not in st.session_state:
    st.session_state.email_sent = False

st.markdown("""
    <h2 style='font-family: Arial;'>🕵️ Fraud Detection Chatbot</h2>
""", unsafe_allow_html=True)

with st.form("user_input_form"):
    st.markdown("<h4 style='font-family: Arial;'>Enter transaction details:</h4>", unsafe_allow_html=True)
    user_input = {}
    for col in features:
        label = col.replace('_', ' ').capitalize()
        if col in ["is_negative_balance_after", "is_late_night", "is_international"]:
            label += " (Yes or No)"
        elif col == "account_age_days":
            label = "Account age (e.g., '12 months' or '2 years')"
        elif col == "transaction_duration":
            label = "Transaction duration (e.g., '2 minutes')"
        elif col == "customer_age":
            label = "Customer age (e.g., '35 years')"
        else:
            label += " (numeric)"
        user_input[col] = st.text_input(label, key=col)

    account_email = st.text_input("User email for fraud alert:")
    submitted = st.form_submit_button("Analyze Transaction")

if submitted:
    user_input["account_age_days"] = parse_account_age(user_input.get("account_age_days", ""))
    user_input["transaction_duration"] = parse_transaction_duration(user_input.get("transaction_duration", ""))
    user_input["customer_age"] = parse_numeric(user_input.get("customer_age", ""))
    user_input["is_late_night"] = parse_yes_no(user_input.get("is_late_night", ""))
    user_input["is_negative_balance_after"] = parse_yes_no(user_input.get("is_negative_balance_after", ""))
    user_input["is_international"] = "Yes" if parse_yes_no(user_input.get("is_international", "")) else "No"

    for k in ["transaction_amount", "balance_before_transaction", "balance_after_transaction", "login_attempts"]:
        user_input[k] = parse_numeric(user_input.get(k, 0))

    input_data = {col: user_input.get(col, 0) for col in features}
    input_df = pd.DataFrame([input_data])
    for col in categorical_cols:
        try:
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
        except:
            input_df[col] = 0

    input_df = input_df.astype(float)
    prediction = model.predict(input_df)[0]
    raw_score = model.decision_function(input_df)[0]

    # Reverse score logic: higher score = more fraud-like
    fraud_score = round(((-raw_score + 0.5) * 100), 2)
    result = "Fraudulent" if prediction == -1 else "Not Fraudulent"
    behavior_cluster = int(kmeans.predict(input_df)[0])

    st.session_state.result_data = {
        "fraud_score": fraud_score,
        "result": result,
        "email": account_email,
        "input_df": input_df,
        "behavior_cluster": behavior_cluster
    }

    st.markdown(f"<h4 style='font-family: Arial;'>Prediction: <strong>{result}</strong></h4>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-family: Arial;'><strong>Fraud Score:</strong> {fraud_score:.2f}%</p>", unsafe_allow_html=True)

    cluster_map = {
        0: "Low-risk cluster. Stable and expected behavior patterns.",
        1: "Moderate-risk cluster. Slight anomalies detected.",
        2: "High-risk cluster. Large or unusual transaction patterns.",
        3: "Erratic behavior cluster. Sparse or suspicious activity history."
    }
    cluster_desc = cluster_map.get(behavior_cluster, "Unknown")
    st.markdown(f"<p style='font-family: Arial;'><strong>Behavioral Cluster:</strong> {behavior_cluster} – {cluster_desc}</p>", unsafe_allow_html=True)

    if result == "Fraudulent" and fraud_score > 75 and not st.session_state.email_sent:
        if st.button("Send Fraud Alert Email"):
            tx_details = "\n".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in user_input.items()])
            sent = send_email_alert(
                to_email=account_email,
                subject="🚨 FRAUD DETECTED - Immediate Action Required",
                message=f"""
Fraudulent transaction detected with {fraud_score:.2f}% confidence.

Behavioral Cluster: {behavior_cluster} - {cluster_desc}

Transaction Details:
{tx_details}

Recommended Actions:
- Verify the transaction
- Contact support if unauthorized
- Monitor account activity immediately
"""
            )
            if sent:
                st.success("✅ Email sent successfully.")
                st.session_state.email_sent = True
