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
    text = text.lower()
    if "month" in text:
        return sanitize_numeric(text) / 12 * 365
    elif "year" in text:
        return sanitize_numeric(text) * 365
    else:
        return sanitize_numeric(text) * 365

def standardize_categoricals(user_input):
    if "is_international" in user_input:
        val = user_input["is_international"].strip().lower()
        user_input["is_international"] = "Yes" if val in ["yes", "y", "true", "1"] else "No"
    return user_input

# ✅ UPDATED GMAIL EMAIL SENDER FUNCTION
def send_email_alert(to_email, subject, message):
    try:
        sender_email = os.getenv("EMAIL_USER")
        sender_password = os.getenv("EMAIL_PASS")
        admin_email = os.getenv("ALERT_ADMIN_EMAIL")
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

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

# Persistent session state
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "result_data" not in st.session_state:
    st.session_state.result_data = {}

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
    user_input["account_age_days"] = parse_account_age(user_input.get("account_age_days", "0"))
    if user_input["account_age_days"] == 0 and user_input.get("transaction_type", "").lower() == "deposit":
        user_input["account_age_days"] = 1

    user_input["transaction_duration"] = sanitize_numeric(user_input.get("transaction_duration", 0)) * 60
    for k in ["transaction_amount", "balance_before_transaction", "balance_after_transaction", "customer_age", "login_attempts"]:
        user_input[k] = sanitize_numeric(user_input.get(k, 0))

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
        0: "Cluster 0: Very low-risk behavior. Transactions are stable and consistent.",
        1: "Cluster 1: Mildly irregular patterns. Occasionally high or delayed transactions.",
        2: "Cluster 2: High alert. This cluster includes frequent or unusually timed high-value actions.",
        3: "Cluster 3: New or minimal history accounts. Risk inferred due to sparse data or erratic behavior."
    }
    st.markdown(f"**Behavioral Cluster:** {d['behavior_cluster']} – {cluster_map.get(d['behavior_cluster'], 'Unknown cluster')} ")

    st.markdown("### Explanation:")
    st.markdown(d['explanation'])

    top_features = X_scored.drop(columns=["anomaly_score", "is_fraud", "behavior_cluster"]).corrwith(X_scored["anomaly_score"]).abs().sort_values(ascending=False).head(10).index
    heatmap_data = X_scored[top_features].copy()
    heatmap_data["anomaly_score"] = X_scored["anomaly_score"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Anomaly Score Heatmap (Top Correlated Features)")
    st.pyplot(fig)

    st.markdown("### Feature Correlation Insights")
    feature_corrs = heatmap_data.corr()['anomaly_score'].drop('anomaly_score')
    for feat, val in feature_corrs.items():
        direction = "increases" if val > 0 else "decreases"
        st.markdown(f"- **{feat.replace('_', ' ').capitalize()}** has a correlation of `{val:.2f}`, meaning higher values {direction} fraud risk.")

    if d['fraud_score'] > 60 and d['email']:
        if st.button("📧 Send Email Alert"):
            tx = "\n".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in d['user_input'].items()])
            email_sent = send_email_alert(
                to_email=d['email'],
                subject="⚠️ Fraud Alert: Suspicious Transaction Detected",
                message=f"""⚠️ A transaction was flagged with a fraud score of {d['fraud_score']}%.

Behavioral Cluster: {d['behavior_cluster']} – {cluster_map.get(d['behavior_cluster'], 'Unknown cluster')}

Reason for Detection:
{d['explanation']}

Transaction Details:
{tx}

Recommended Actions:
- Verify this transaction
- Contact your bank if unauthorized
- Monitor account activity for anomalies."""
            )
            if email_sent:
                st.success("✅ Alert sent to account owner and admin.")
