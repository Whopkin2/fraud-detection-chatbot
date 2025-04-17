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

# Feature Engineering
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
        st.error(f"❌ Email alert failed: {e}")
        return False

if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "result_data" not in st.session_state:
    st.session_state.result_data = {}
if "email_sent" not in st.session_state:
    st.session_state.email_sent = False

st.markdown("## 🕵️ <span style='font-family: Arial;'>Fraud Detection Chatbot</span>", unsafe_allow_html=True)

with st.form("user_input_form"):
    st.markdown("### <span style='font-family: Arial;'>Enter transaction data:</span>", unsafe_allow_html=True)
    user_input = {}
    for col in features:
        label = col.replace('_', ' ').capitalize()
        if col in ["is_negative_balance_after", "is_late_night"]:
            label += " (Yes or No)"
            user_input[col] = st.selectbox(label, ["Yes", "No"], key=col)
        elif col == "account_age_days":
            label = "Account age (e.g., '12 months' or '2 years')"
            user_input[col] = st.text_input(label, key=col)
        elif col == "transaction_duration":
            label = "Transaction duration (e.g., '3 minutes' or '2 hours')"
            user_input[col] = st.text_input(label, key=col)
        elif col == "customer_age":
            label = "Customer age (e.g., '24 years')"
            user_input[col] = st.text_input(label, key=col)
        elif col == "is_international":
            label += " (categorical)"
            user_input[col] = st.selectbox(label, ["Yes", "No"], key=col)
        elif col == "transaction_method":
            label += " (categorical)"
            user_input[col] = st.selectbox(label, ["ATM", "Online", "POS", "Mobile", "Wire"], key=col)
        elif col == "time_of_day":
            label += " (categorical)"
            user_input[col] = st.selectbox(label, ["Morning", "Afternoon", "Evening", "Night"], key=col)
        else:
            label += " (numeric)"
            user_input[col] = st.text_input(label, key=col)

    account_owner_email = st.text_input("Account owner's email (for alert):")
    submitted = st.form_submit_button("Analyze Transaction")

if submitted:
    user_input = standardize_categoricals(user_input)
    user_input["account_age_days"] = parse_account_age(user_input.get("account_age_days", "1 year"))
    user_input["transaction_duration"] = parse_transaction_duration(user_input.get("transaction_duration", "1 minute"))
    user_input["customer_age"] = parse_customer_age(user_input.get("customer_age", "18"))

    for k in ["transaction_amount", "balance_before_transaction", "balance_after_transaction", "login_attempts"]:
        user_input[k] = sanitize_numeric(user_input.get(k, "0"))

    user_input["is_negative_balance_after"] = parse_yes_no(user_input.get("is_negative_balance_after", "No"))
    user_input["is_late_night"] = parse_yes_no(user_input.get("is_late_night", "No"))

    full_row = {col: user_input.get(col, 0 if col not in categorical_cols else "Unknown") for col in features}
    input_df = pd.DataFrame([full_row])

    for col in categorical_cols:
        try:
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
        except:
            fallback = label_encoders[col].classes_[0]
            input_df[col] = label_encoders[col].transform([fallback])[0]

    input_df = input_df.astype(float).reindex(columns=X.columns, fill_value=0)

    prediction = isolation_model.predict(input_df)[0]
    raw_score = isolation_model.decision_function(input_df)[0]
    confidence_score = round(((-raw_score + 0.5) / 1.0) * 100, 2)
    confidence_score = max(0.0, min(confidence_score, 100.0))
    result = "Fraudulent" if prediction == -1 else "Not Fraudulent"

    rating = 1.0
    if user_input["account_age_days"] < 90: rating += 1.0
    if user_input["login_attempts"] > 3: rating += 0.5
    if user_input["transaction_amount"] > 5000: rating += 1.0
    if user_input["is_late_night"]: rating += 0.5
    if user_input["transaction_method"] in ["Online", "Mobile"]: rating += 0.5
    if prediction == -1: rating += 0.5
    rating = min(5.0, round(rating, 1))

    explanation_lines = [
        f"- Transaction Amount: ${user_input['transaction_amount']} – higher amounts are often suspicious.",
        f"- Account Age: {user_input['account_age_days']} days – newer accounts tend to have higher risk.",
        f"- Login Attempts: {user_input['login_attempts']} – excessive login attempts raise red flags.",
        f"- Time of Day: {user_input['time_of_day']} – late hours can indicate attempts to avoid detection.",
        f"- Transaction Method: {user_input['transaction_method']} – remote methods can mask identity."
    ]

    prompt = f"""
Given the transaction data: {user_input},\nPredicted: {result} with a confidence score of {confidence_score}%,\nBehavioral Risk Rating: {rating}/5\nExplain these findings to the user in layman's terms.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful fraud risk advisor who explains AI-based anomaly detection decisions."},
            {"role": "user", "content": prompt}
        ]
    )
    explanation = response.choices[0].message.content

    st.session_state.submitted = True
    st.session_state.result_data = {
        "user_input": user_input,
        "result": result,
        "confidence_score": confidence_score,
        "behavior_rating": rating,
        "email": account_owner_email,
        "input_df": input_df,
        "explanation": explanation,
        "anomaly_insights": explanation_lines
    }

if st.session_state.submitted:
    d = st.session_state.result_data
    st.markdown(f"### Prediction: **{d['result']}**")
    st.markdown(f"**Confidence Level:** {d['confidence_score']}% Confident")

    if "behavior_rating" in d:
        st.markdown(f"**🧠 Behavioral Risk Rating:** {d['behavior_rating']} / 5")

    st.markdown("### Explanation:")
    st.markdown(d['explanation'])

    st.markdown("### 🔍 Feature Highlights Contributing to Detection:")
    for insight in d.get('anomaly_insights', []):
        st.markdown(insight)

    if d['result'] == "Fraudulent" and d['confidence_score'] >= 50 and d['email'] and not st.session_state.email_sent:
        if st.button("📧 Send Fraud Alert Email"):
            tx = "\n".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in d['user_input'].items()])
            email_sent = send_email_alert(
                to_email=d['email'],
                subject="🚨 FRAUD ALERT – Suspicious Transaction Detected",
                message=f"""A transaction was flagged with a **confidence level of {d['confidence_score']}%**.

Behavioral Risk Rating: {d['behavior_rating']} / 5

Transaction Details:
{tx}

Recommended Actions:
- Verify this transaction
- Contact your bank if unauthorized
- Monitor account activity immediately."""
            )
            if email_sent:
                st.success("✅ Alert sent to account owner and admin.")
                st.session_state.email_sent = True
            else:
                st.error("❌ Email failed to send.")
