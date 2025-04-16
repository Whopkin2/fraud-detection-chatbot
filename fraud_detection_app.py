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

# Load API key and email config from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()

DATA_PATH = "Banking Transactions Data For Fraud.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)

data = load_data()

# Drop unnecessary columns
columns_to_drop = ["device_id", "transaction_id"]
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# Encode categorical columns
categorical_cols = data.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Clean training data (X)
X = data.copy()

# Fit Isolation Forest before modifying data
isolation_model = IsolationForest(contamination=0.05, random_state=42)
isolation_model.fit(X)

# Duplicate for analysis
X_scored = X.copy()
X_scored["anomaly_score"] = isolation_model.decision_function(X_scored)
X_scored["is_fraud"] = isolation_model.predict(X_scored)

# Behavioral clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
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
            st.success("🚨 Email alert sent!")
    except Exception as e:
        st.error(f"Email alert failed: {e}")

def predict_fraud(user_input):
    user_input = standardize_categoricals(user_input)

    if "account_age_days" in user_input:
        user_input["account_age_days"] = sanitize_numeric(user_input["account_age_days"]) * 365
    if "transaction_duration" in user_input:
        user_input["transaction_duration"] = sanitize_numeric(user_input["transaction_duration"]) * 60

    for key in ["transaction_amount", "balance_before_transaction", "balance_after_transaction", "customer_age", "login_attempts"]:
        if key in user_input:
            user_input[key] = sanitize_numeric(user_input[key])

    full_row = {col: user_input.get(col, 0 if col not in categorical_cols else "Unknown") for col in X.columns}
    input_df = pd.DataFrame([full_row])

    for col in input_df.columns:
        if col in categorical_cols:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
            except ValueError:
                fallback = label_encoders[col].classes_[0]
                input_df[col] = [label_encoders[col].transform([fallback])[0]]
        elif col == "is_international":
            val = str(input_df[col].values[0]).strip().lower()
            input_df[col] = 1 if val in ["yes", "y", "true", "1"] else 0

    input_df = input_df.astype(float)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    prediction = isolation_model.predict(input_df)[0]
    fraud_score = round(abs(isolation_model.decision_function(input_df)[0]), 2)
    behavior_cluster = kmeans.predict(input_df)[0]
    result = 1 if prediction == -1 else 0
    return result, fraud_score, behavior_cluster

questions = [
    "Transaction Amount:",
    "Transaction Type (e.g., Purchase, Transfer, Withdrawal):",
    "Account Age (Years):",
    "Is this International? (Yes/No):",
    "Time of Day:",
    "Customer Age:",
    "Branch Code (BR001–BR004):",
    "Transaction Method (e.g., Online, In-Person, Mobile):",
    "Balance Before Transaction:",
    "Balance After Transaction:",
    "Login Attempts:",
    "Transaction Duration (minutes):"
]

keys = [
    "transaction_amount", "transaction_type", "account_age_days",
    "is_international", "time_of_day", "customer_age", "branch_code",
    "transaction_method", "balance_before_transaction",
    "balance_after_transaction", "login_attempts", "transaction_duration"
]

if "question_index" not in st.session_state:
    st.session_state.question_index = 0
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

st.markdown("## 🕵️ Fraud Detection Chatbot")
st.markdown("Enter values for a sample transaction. The bot will flag unusual patterns based on prior data:")

for message in st.session_state.chat_log:
    st.markdown(message, unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    if st.session_state.question_index < len(questions):
        prompt_text = questions[st.session_state.question_index]

        if "Transaction Type" in prompt_text:
            user_input = st.selectbox(prompt_text, ["Purchase", "Transfer", "Withdrawal", "Deposit"])
        elif "Transaction Method" in prompt_text:
            user_input = st.selectbox(prompt_text, ["Online", "In-Person", "Mobile", "ATM"])
        elif "Is this International" in prompt_text:
            user_input = st.selectbox(prompt_text, ["Yes", "No"])
        else:
            user_input = st.text_input(prompt_text)

        submitted = st.form_submit_button("Send")
        if submitted and user_input:
            key = keys[st.session_state.question_index]
            st.session_state.user_answers[key] = user_input
            st.session_state.chat_log.append(f"<b>You:</b> {user_input}")
            st.session_state.question_index += 1
            if st.session_state.question_index < len(questions):
                next_q = questions[st.session_state.question_index]
                st.session_state.chat_log.append(f"<b>FraudBot:</b> {next_q}")
            st.rerun()
    else:
        prediction, fraud_score, behavior_cluster = predict_fraud(st.session_state.user_answers)
        result = "🚨 Fraudulent" if prediction == 1 else "✅ Not Fraudulent"

        if prediction == 1:
            send_email_alert(
                to_email="your_recipient@example.com",
                subject="⚠️ Fraud Alert: Suspicious Transaction Detected",
                message=f"A potentially fraudulent transaction was flagged:\n\n{st.session_state.user_answers}\n\nFraud Score: {fraud_score}"
            )

        prompt = (
            f"Given the transaction data: {st.session_state.user_answers},\n"
            f"and a model that flags this transaction as {result} with fraud score {fraud_score},\n"
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

        st.markdown(f"### 🔍 Prediction: {result}")
        st.markdown(f"**Fraud Score:** {fraud_score}")
        st.markdown(f"**Behavioral Cluster:** {behavior_cluster}")
        st.markdown("---")
        st.markdown("### 💡 Risk Assessment:")
        st.markdown(explanation)

        # Anomaly Heatmap
        top_features = X_scored.drop(columns=["anomaly_score", "is_fraud", "behavior_cluster"]).corrwith(X_scored["anomaly_score"]).abs().sort_values(ascending=False).head(10).index
        heatmap_data = X_scored[top_features].copy()
        heatmap_data["anomaly_score"] = X_scored["anomaly_score"]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("🔍 Anomaly Score Heatmap (Top Correlated Features)")
        st.pyplot(fig)
