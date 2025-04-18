import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import numpy as np
import smtplib
from email.mime.text import MIMEText
import re
import matplotlib.pyplot as plt
import seaborn as sns

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

isolation_model = IsolationForest(contamination=0.05, n_estimators=200, max_features=0.9, random_state=42)
isolation_model.fit(X)

def sanitize_numeric(value):
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").strip()
    try:
        return float(value)
    except:
        return 0.0

def extract_number(text):
    try:
        return float(re.search(r"\d*\.?\d+", str(text)).group())
    except:
        return 0.0

def parse_account_age(text):
    text = str(text).lower()
    num = extract_number(text)
    if "month" in text:
        return num * 30
    elif "year" in text:
        return num * 365
    return num

def parse_transaction_duration(text):
    text = str(text).lower()
    num = extract_number(text)
    if "second" in text:
        return num / 60
    elif "hour" in text:
        return num * 60
    return num

def parse_customer_age(text):
    return extract_number(text)

def parse_yes_no(value):
    return 1 if str(value).strip().lower() in ['yes', 'y', 'true', '1'] else 0

def standardize_categoricals(user_input):
    if "is_international" in user_input:
        val = user_input["is_international"].strip().lower()
        user_input["is_international"] = "Yes" if val in ["yes", "y", "true", "1"] else "No"
    return user_input

def calculate_confidence_from_rating(score):
    if score >= 5.0:
        return 95.0
    elif score >= 4.0:
        return 85.0
    elif score >= 3.5:
        return 75.0
    elif score >= 2.5:
        return 60.0
    elif score >= 2.0:
        return 50.0
    elif score >= 1.0:
        return 30.0
    else:
        return 10.0

def compute_behavioral_risk_score(user):
    score = 0
    if user["account_age_days"] < 90:
        score += 1.0
    else:
        score -= 1.0
    if user["login_attempts"] > 3:
        score += 0.5
    else:
        score -= 0.5
    if user["transaction_amount"] > 5000:
        score += 1.0
    else:
        score -= 1.0
    if user["is_late_night"] == 1:
        score += 0.5
    else:
        score -= 0.5
    if user["transaction_method"] in ["Online", "Mobile", "Wire"]:
        score += 0.5
    else:
        score -= 0.5
    if user["is_international"] == "Yes":
        score += 0.5
    else:
        score -= 0.5
    if user["is_negative_balance_after"] == 1:
        score += 0.5
    else:
        score -= 0.5
    if user["transaction_duration"] <= 2:
        score += 0.5
    else:
        score -= 0.5
    if user["customer_age"] < 24:
        score += 0.5
    else:
        score -= 0.5
    return max(0, min(5, round(score, 2)))

def send_email_alert(to_email, subject, message):
    try:
        sender_email = os.getenv("ALERT_SENDER_EMAIL")
        sender_password = os.getenv("EMAIL_PASS")
        admin_email = os.getenv("ALERT_ADMIN_EMAIL")
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        if not sender_email or not sender_password or not admin_email:
            raise Exception("Missing environment variables for email.")

        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [to_email, admin_email], msg.as_string())

        return True
    except Exception as e:
        st.error(f"\u274c Email alert failed: {e}")
        return False

if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "result_data" not in st.session_state:
    st.session_state.result_data = {}
if "email_sent" not in st.session_state:
    st.session_state.email_sent = False

# GPT logic to run after result calculated
if st.session_state.submitted:
    d = st.session_state.result_data

    explanation_lines = [
        f"- Transaction Amount: ${d['user_input']['transaction_amount']} – higher amounts are often suspicious.",
        f"- Account Age: {d['user_input']['account_age_days']} days – newer accounts tend to have higher risk.",
        f"- Login Attempts: {d['user_input']['login_attempts']} – excessive login attempts raise red flags.",
        f"- Time of Day: {d['user_input']['time_of_day']} – late hours can indicate attempts to avoid detection.",
        f"- Transaction Method: {d['user_input']['transaction_method']} – remote methods can mask identity."
    ]

    prompt = f"""
Given the transaction data: {d['user_input']},\nPredicted: {d['result']} with a confidence score of {d['confidence_score']}%,\nBehavioral Risk Rating: {d['behavior_rating']}/5\nExplain these findings to the user in layman's terms.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful fraud risk advisor who explains AI-based anomaly detection decisions."},
                {"role": "user", "content": prompt}
            ]
        )
        d['explanation'] = response.choices[0].message.content
        d['anomaly_insights'] = explanation_lines
    except Exception as e:
        d['explanation'] = "Unable to generate explanation."
        d['anomaly_insights'] = explanation_lines
