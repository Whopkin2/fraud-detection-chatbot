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

    rating = compute_behavioral_risk_score(user_input)
    confidence_score = calculate_confidence_from_rating(rating)

    prediction = isolation_model.predict(input_df)[0]
    result = "Fraudulent" if prediction == -1 else "Not Fraudulent"

    st.session_state.submitted = True
    st.session_state.result_data = {
        "user_input": user_input,
        "result": result,
        "confidence_score": confidence_score,
        "behavior_rating": rating,
        "email": account_owner_email,
        "input_df": input_df
    }

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
    st.markdown("### Explanation:")
    st.markdown(d.get('explanation', 'Explanation not available.'))

    st.markdown("### 🔍 Feature Highlights Contributing to Detection:")
    for insight in d.get('anomaly_insights', []):
        st.markdown(insight)

    st.markdown("### 📊 Adjusted Anomaly Heatmap (Fraud Risk Based):")
    fig, ax = plt.subplots(figsize=(10, 6))
    heat_data = d['input_df'].T
    sns.heatmap(heat_data, annot=True, cmap="Reds", fmt=".2f", ax=ax, cbar_kws={'label': 'Feature Value'})
    st.pyplot(fig)

    st.markdown("**🔍 Heatmap Explanation:**")
    for feature, value in d['input_df'].iloc[0].items():
        explanation = ""
        if feature == "transaction_amount":
            explanation = "Higher values here often flag potential fraud due to large transfers or withdrawals."
        elif feature == "account_age_days":
            explanation = "Lower account age (fewer days active) is riskier and flagged more often."
        elif feature == "login_attempts":
            explanation = "An unusually high number of login attempts may imply unauthorized access attempts."
        elif feature == "transaction_duration":
            explanation = "Very short transaction times may suggest automated fraud or scripted behavior."
        elif feature == "is_late_night":
            explanation = "Late-night activity has a higher correlation with fraudulent behavior in historical data."
        elif feature == "is_negative_balance_after":
            explanation = "Ending in a negative balance often implies misuse or overdraft attempts."
        else:
            explanation = "This value was scored based on deviation from normal behavior learned by the model."
        st.markdown(f"- **{feature.replace('_', ' ').capitalize()}**: `{value:.2f}` → {explanation}")

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
