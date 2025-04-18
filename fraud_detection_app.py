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

def calculate_confidence_from_rating(score):
    # We want confidence to be lowest at 2.5 and highest at 0 or 5
    distance_from_center = abs(score - 2.5)

    # Scale distance from 0–2.5 into a 50%–100% confidence range
    confidence = 50.0 + (distance_from_center / 2.5) * 50.0

    return round(confidence, 2)

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

    if user["transaction_amount"] > 10000:
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

st.markdown("## 🕵️ <span style='font-family: Arial;'>Fraud Detection AI Tool</span>", unsafe_allow_html=True)

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

if st.session_state.submitted:
    d = st.session_state.result_data
    st.markdown(f"### Prediction: **{d['result']}**")
    st.markdown(f"**Confidence Level:** {d['confidence_score']}% Confident")

    st.markdown("### 🧠 Behavioral Risk Rating Breakdown")
    score = d['behavior_rating']
    user = d['user_input']
    score_factors = []

    score_factors.append(("Account Age", +1.0 if user["account_age_days"] < 90 else -1.0, "Account is new" if user["account_age_days"] < 90 else "Account is established"))
    score_factors.append(("Login Attempts", +0.5 if user["login_attempts"] > 3 else -0.5, "Too many login attempts" if user["login_attempts"] > 3 else "Login count is normal"))
    score_factors.append(("Transaction Amount", +1.0 if user["transaction_amount"] > 5000 else -1.0, "Large transaction amount" if user["transaction_amount"] > 5000 else "Amount is modest"))
    score_factors.append(("Time of Day", +0.5 if user["is_late_night"] == 1 else -0.5, "Suspicious late-night timing" if user["is_late_night"] == 1 else "Normal hours"))
    score_factors.append(("Method", +0.5 if user["transaction_method"] in ["Online", "Mobile", "Wire"] else -0.5, "Remote transaction method" if user["transaction_method"] in ["Online", "Mobile", "Wire"] else "In-person method"))
    score_factors.append(("International", +0.5 if user["is_international"] == "Yes" else -0.5, "International transaction" if user["is_international"] == "Yes" else "Domestic transaction"))
    score_factors.append(("Negative Balance", +0.5 if user["is_negative_balance_after"] == 1 else -0.5, "Ends in negative balance" if user["is_negative_balance_after"] == 1 else "Balance is sufficient"))
    score_factors.append(("Short Duration", +0.5 if user["transaction_duration"] <= 2 else -0.5, "Suspiciously fast transaction" if user["transaction_duration"] <= 2 else "Normal duration"))
    score_factors.append(("Young Age", +0.5 if user["customer_age"] < 24 else -0.5, "Very young customer" if user["customer_age"] < 24 else "Customer age is mature"))

    st.markdown(f"**Total Behavioral Risk Rating: {score} / 5**")
    for factor, impact, reason in score_factors:
        sign = "+" if impact > 0 else "–"
        st.markdown(f"- **{factor}**: {sign}{abs(impact)} → _{reason}_")

    if score >= 4.0:
        summary = "This transaction shows multiple high-risk traits. Please investigate urgently."
    elif score >= 2.5:
        summary = "There are moderate risk indicators present. Further review is recommended."
    else:
        summary = "Low behavioral risk detected. Transaction appears typical."

    st.markdown(f"📌 **Summary**: {summary}")

    # ✅ Cleaned GPT explanation
    explanation = d.get("explanation", "Explanation not available.")
    explanation_cleaned = explanation.replace("\n", " ").replace("  ", " ").strip()
    st.markdown("### 🧠 Explanation:")
    st.markdown(
        f"<div style='font-family: Arial; font-size: 16px; line-height: 1.6;'>{explanation_cleaned}</div>",
        unsafe_allow_html=True
    )

    # ✅ Anomaly insights
    st.markdown("### 🔍 Feature Highlights Contributing to Detection:")
    for insight in d.get('anomaly_insights', []):
        st.markdown(insight)

    # ✅ Heatmap + Summary
    st.markdown("### 📊 Adjusted Anomaly Heatmap (Fraud Risk Based):")

    # Same heatmap logic already in your code
    heatmap_data = []
    annotations = []
    summary_lines = []

    for feature, (condition, pos_score, pos_desc, neg_score, neg_desc) in risk_logic.items():
        score = 3.0 if "+" in pos_score and condition else 1.0
        label = pos_score if condition else neg_score
        desc = pos_desc if condition else neg_desc
        status = "🔴 High Risk" if score == 3.0 else "🔵 Low Risk"

        heatmap_data.append((feature, score, label))
        annotations.append(desc)
        summary_lines.append(f"- **{feature}** → {desc} → **{label}** ({status})")

    heatmap_df = pd.DataFrame(heatmap_data, columns=["Feature", "RiskScore", "Label"]).set_index("Feature")
    heatmap_df["Explanation"] = annotations

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_df[["RiskScore"]],
        annot=heatmap_df[["Label"]],
        fmt="",
        cmap="RdBu_r",
        center=2.0,
        linewidths=0.5,
        cbar_kws={"label": "Fraud Likelihood Score"},
        ax=ax
    )
    plt.title("Adjusted Anomaly Heatmap (Fraud Risk Based)", fontsize=14)
    st.pyplot(fig)

    st.markdown("### 📋 Heatmap Summary Explanation:")
    for line in summary_lines:
        st.markdown(line)

    # ✅ Email button section
    if d['result'] == "Fraudulent" and d['confidence_score'] >= 50 and d.get('email') and not st.session_state.email_sent:
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
