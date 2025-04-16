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
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

DATA_PATH = "Banking Transactions Data For Fraud.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)

data = load_data()
data = data.drop(columns=["transaction_id", "device_id", "branch_code"])

categorical_cols = data.select_dtypes(include=['object', 'bool']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

X = data.copy()
isolation_model = IsolationForest(contamination=0.05, random_state=42)
isolation_model.fit(X)

X_scored = X.copy()
X_scored["anomaly_score"] = isolation_model.decision_function(X)
X_scored["is_fraud"] = isolation_model.predict(X)

kmeans = KMeans(n_clusters=4, random_state=42)
X_scored["behavior_cluster"] = kmeans.fit_predict(X)

def sanitize_numeric(value):
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").strip()
    try:
        return float(value)
    except:
        return 0.0

def parse_account_age(text):
    try:
        text = text.lower()
        if "month" in text:
            return sanitize_numeric(text) / 12 * 365
        elif "year" in text:
            return sanitize_numeric(text) * 365
        else:
            return sanitize_numeric(text) * 365  # assume years if no unit
    except:
        return 1  # default to 1 day if missing or error

def standardize_categoricals(user_input):
    if "is_international" in user_input:
        val = user_input["is_international"].strip().lower()
        user_input["is_international"] = "Yes" if val in ["yes", "y", "true", "1"] else "No"
    return user_input

def send_email_alert(to_email, subject, message):
    try:
        sender_email = os.getenv("EMAIL_USER")
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
            server.set_debuglevel(1)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [to_email, admin_email], msg.as_string())

        return True
    except Exception as e:
        st.error(f"❌ Email alert failed: {e}")
        return False

# Session state
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "result_data" not in st.session_state:
    st.session_state.result_data = {}
if "email_sent" not in st.session_state:
    st.session_state.email_sent = False

st.markdown("## 🕵️ Fraud Detection Chatbot")

with st.form("user_input_form"):
    st.markdown("### Enter transaction data:")
    user_input = {}
    for col in X.columns:
        label = col.replace('_', ' ').capitalize()
        if col in categorical_cols:
            if col == "transaction_type":
                label += " (e.g., payment / transfer / withdrawal / deposit)"
            elif col == "time_of_day":
                label += " (e.g., morning / afternoon / evening / night)"
            elif col == "transaction_method":
                label += " (e.g., online / ATM / swipe / in-person)"
            elif col == "is_international":
                label += " (Yes or No)"
        else:
            if col == "account_age_days":
                label = "Account age (e.g., '12 months' or '2 years'):"
            elif col == "transaction_duration":
                label = "Transaction duration (in minutes):"
            else:
                label += " (numeric)"
        user_input[col] = st.text_input(label, key=col)

    account_owner_email = st.text_input("Account owner's email (for alert):")
    submitted = st.form_submit_button("Analyze Transaction")

if submitted:
    user_input = standardize_categoricals(user_input)
    user_input["account_age_days"] = parse_account_age(user_input.get("account_age_days", "1 year"))
    user_input["transaction_duration"] = sanitize_numeric(user_input.get("transaction_duration", "1")) * 60
    for k in ["transaction_amount", "balance_before_transaction", "balance_after_transaction", "customer_age", "login_attempts"]:
        user_input[k] = sanitize_numeric(user_input.get(k, "0"))

    full_row = {col: user_input.get(col, 0 if col not in categorical_cols else "Unknown") for col in X.columns}
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
    fraud_score = round((1 - raw_score) * 100, 2)
    behavior_cluster = int(kmeans.predict(input_df)[0])
    result = "Fraudulent" if prediction == -1 else "Not Fraudulent"

    prompt = f"""
Given the transaction data: {user_input},
and a model that flagged it as {result} with fraud score {fraud_score},
evaluate the transaction in detail. Include account age, login attempts, transaction type and duration,
and explain how these factors influence the model's decision.
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
        "fraud_score": fraud_score,
        "behavior_cluster": behavior_cluster,
        "explanation": explanation,
        "email": account_owner_email
    }

if st.session_state.submitted:
    d = st.session_state.result_data
    st.markdown(f"### Prediction: **{d['result']}**")
    st.markdown(f"**Fraud Score:** {d['fraud_score']}%")

    cluster_map = {
        0: "Low-risk cluster with consistent behavior and established transaction patterns.",
        1: "Mildly irregular cluster — moderate risk with some timing/amount deviations.",
        2: "High-alert cluster with frequent large or off-hour transactions.",
        3: "Erratic behavior cluster. Sparse history or unusual patterns — often seen in new or suspicious accounts."
    }

    if d['result'] == "Fraudulent" and d['fraud_score'] > 60:
        cluster_explanation = (
            f"{cluster_map.get(d['behavior_cluster'], 'Unknown cluster')} "
            f"However, this transaction was flagged as fraudulent, indicating a risk spike not typical for this profile."
        )
    else:
        cluster_explanation = cluster_map.get(d['behavior_cluster'], 'Unknown cluster')

    st.markdown(f"**Behavioral Cluster:** {d['behavior_cluster']} – {cluster_explanation}")
    st.markdown("### Explanation:")
    st.markdown(d['explanation'])

    st.markdown("### 🔍 Key Feature Values for This Transaction:")
    top_input_features = input_df.iloc[0].sort_values(ascending=False).head(5)
    for feat, val in top_input_features.items():
        st.markdown(f"- **{feat.replace('_', ' ').capitalize()}**: `{val:.2f}`")

    if d['fraud_score'] > 60 and d['email'] and not st.session_state.email_sent:
        tx = "\n".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in d['user_input'].items()])
        email_sent = send_email_alert(
            to_email=d['email'],
            subject="WARNING: Fraud Alert – Suspicious Transaction Detected",
            message=f"""A transaction was flagged with a fraud score of {d['fraud_score']}%.

Behavioral Cluster: {d['behavior_cluster']} – {cluster_explanation}

Reason for Detection:
{d['explanation']}

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
