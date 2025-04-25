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
    score_factors = [
        ("Account Age", 1.0, user["account_age_days"] < 90, "Account is new", "Account is established"),
        ("Login Attempts", 0.75, user["login_attempts"] > 3, "Too many login attempts", "Login count is normal"),
        ("Transaction Amount", 1.0, user["transaction_amount"] > 10000, "Large transaction amount", "Amount is modest"),
        ("Time of Day", 0.5, user["is_late_night"] == 1, "Suspicious late-night timing", "Normal hours"),
        ("Method", 0.25, user["transaction_method"] in ["Online", "Mobile", "Wire"], "Remote transaction method", "In-person method"),
        ("International", 0.75, user["is_international"] == "Yes", "International transaction", "Domestic transaction"),
        ("Negative Balance", 0.25, user["is_negative_balance_after"] == 1, "Ends in negative balance", "Balance is sufficient"),
        ("Short Duration", 0.25, user["transaction_duration"] <= 2, "Suspiciously fast transaction", "Normal duration"),
        ("Young Age", 0.25, user["customer_age"] < 24, "Very young customer", "Customer age is mature")
    ]

    rating = 0
    for _, weight, condition, _, _ in score_factors:
        impact = weight if condition else -weight
        rating += impact
        rating = max(0, min(5, rating))  # Clamp after each impact

    return round(rating, 2)

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

    # 1. Compute behavioral risk rating
    score_factors = [
        ("Account Age", +1.0 if user_input["account_age_days"] < 90 else -1.0, "Account is new" if user_input["account_age_days"] < 90 else "Account is established"),
        ("Login Attempts", +0.75 if user_input["login_attempts"] > 3 else -0.75, "Too many login attempts" if user_input["login_attempts"] > 3 else "Login count is normal"),
        ("Transaction Amount", +1.0 if user_input["transaction_amount"] > 10000 else -1.0, "Large transaction amount" if user_input["transaction_amount"] > 10000 else "Amount is modest"),
        ("Time of Day", +0.5 if user_input["is_late_night"] == 1 else -0.5, "Suspicious late-night timing" if user_input["is_late_night"] == 1 else "Normal hours"),
        ("Method", +0.25 if user_input["transaction_method"] in ["Online", "Mobile", "Wire"] else -0.25, "Remote transaction method" if user_input["transaction_method"] in ["Online", "Mobile", "Wire"] else "In-person method"),
        ("International", +0.75 if user_input["is_international"] == "Yes" else -0.75, "International transaction" if user_input["is_international"] == "Yes" else "Domestic transaction"),
        ("Negative Balance", +0.25 if user_input["is_negative_balance_after"] == 1 else -0.25, "Ends in negative balance" if user_input["is_negative_balance_after"] == 1 else "Balance is sufficient"),
        ("Short Duration", +0.25 if user_input["transaction_duration"] <= 2 else -0.25, "Suspiciously fast transaction" if user_input["transaction_duration"] <= 2 else "Normal duration"),
        ("Young Age", +0.25 if user_input["customer_age"] < 24 else -0.25, "Very young customer" if user_input["customer_age"] < 24 else "Customer age is mature")
    ]

    # 1. Compute behavioral risk rating
    rating = compute_behavioral_risk_score(user_input)

    # 2. Compute confidence score from rating
    confidence_score = calculate_confidence_from_rating(rating)

    # 3. Get model prediction
    prediction = isolation_model.predict(input_df)[0]
    if rating >= 3.0:
        result = "Fraudulent"
    elif rating < 2.0:
        result = "Not Fraudulent"
    else:
        result = "Fraudulent" if prediction == -1 else "Not Fraudulent"

    # 4. Generate GPT explanation
    explanation_prompt = f"""
    Given the transaction data: {user_input},
    Predicted: {result} with a confidence score of {confidence_score}%,
    Behavioral Risk Rating: {rating}/5
    Explain these findings to the user in layman's terms.
    """

    try:
        gpt_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful fraud risk advisor who explains AI-based anomaly detection decisions."},
                {"role": "user", "content": explanation_prompt}
            ]
        )
        explanation = gpt_response.choices[0].message.content
    except Exception as e:
        explanation = f"❌ GPT explanation failed: {e}"

    # 5. Store everything in session state
    st.session_state.submitted = True
    st.session_state.result_data = {
        "user_input": user_input,
        "result": result,
        "confidence_score": confidence_score,
        "behavior_rating": rating,
        "email": account_owner_email.strip() if account_owner_email.strip() else None,
        "input_df": input_df,
        "explanation": explanation,
        "anomaly_insights": []  # You can populate this if you want more GPT-generated highlights
    }

if st.session_state.submitted:
    d = st.session_state.result_data
    st.markdown(f"<h3 style='font-family: Arial;'>Prediction: <b>{d['result']}</b></h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-family: Arial;'><b>Confidence Level:</b> {d['confidence_score']}% Confident</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='font-family: Arial;'>\U0001f9e0 Behavioral Risk Rating Breakdown</h3>", unsafe_allow_html=True)
    user = st.session_state.result_data['user_input']

    score_factors = [
        ("Account Age", 1.0, user["account_age_days"] < 90, "Account is new", "Account is established"),
        ("Login Attempts", 0.75, user["login_attempts"] > 3, "Too many login attempts", "Login count is normal"),
        ("Transaction Amount", 1.0, user["transaction_amount"] > 10000, "Large transaction amount", "Amount is modest"),
        ("Time of Day", 0.5, user["is_late_night"] == 1, "Suspicious late-night timing", "Normal hours"),
        ("Method", 0.25, user["transaction_method"] in ["Online", "Mobile", "Wire"], "Remote transaction method", "In-person method"),
        ("International", 0.75, user["is_international"] == "Yes", "International transaction", "Domestic transaction"),
        ("Negative Balance", 0.25, user["is_negative_balance_after"] == 1, "Ends in negative balance", "Balance is sufficient"),
        ("Short Duration", 0.25, user["transaction_duration"] <= 2, "Suspiciously fast transaction", "Normal duration"),
        ("Young Age", 0.25, user["customer_age"] < 24, "Very young customer", "Customer age is mature")
    ]

    rating = d['behavior_rating']  # Already calculated during form submit

    for factor, weight, condition, pos_desc, neg_desc in score_factors:
        impact = weight if condition else -weight
        reason = pos_desc if condition else neg_desc
        sign = "+" if impact > 0 else "–"
        st.markdown(f"- **{factor}**: {sign}{abs(weight)} → _{reason}_")

    st.markdown(f"**Total Behavioral Risk Rating: {rating} / 5**")

    if rating >= 4.0:
        summary = "This transaction shows multiple high-risk traits. Please investigate urgently."
    elif rating >= 2.5:
        summary = "There are moderate risk indicators present. Further review is recommended."
    else:
        summary = "Low behavioral risk detected. Transaction appears typical."

    st.markdown(f"📌 **Summary**: {summary}")
    st.markdown("<h3 style='font-family: Arial;'>🧠 Explanation:</h3>", unsafe_allow_html=True)
    st.markdown(
        f"<pre style='font-family: Arial; font-size: 16px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; border: none; background: none;'>{d.get('explanation', 'Explanation not available.')}</pre>",
        unsafe_allow_html=True
    )

    st.markdown("<h3 style='font-family: Arial;'>🔍 Feature Highlights Contributing to Detection:</h3>", unsafe_allow_html=True)
    for insight in d.get('anomaly_insights', []):
        st.markdown(insight)

    st.markdown("<h3 style='font-family: Arial;'>📊 Adjusted Anomaly Heatmap (Fraud Risk Based):</h3>", unsafe_allow_html=True)

    risk_logic = {
        "Account Age": (user["account_age_days"] < 90, "+1.0", "Account is new", "-1.0", "Account is established"),
        "Login Attempts": (user["login_attempts"] > 3, "+0.75", "Too many login attempts", "-0.75", "Login count is normal"),
        "Transaction Amount": (user["transaction_amount"] > 10000, "+1.0", "Large transaction amount", "-1.0", "Amount is modest"),
        "Time of Day": (user["is_late_night"] == 1, "+0.5", "Suspicious late-night timing", "-0.5", "Normal hours"),
        "Method": (user["transaction_method"] in ["Online", "Mobile", "Wire"], "+0.25", "Remote transaction method", "-0.25", "In-person method"),
        "International": (user["is_international"] == "Yes", "+0.75", "International transaction", "-0.75", "Domestic transaction"),
        "Negative Balance": (user["is_negative_balance_after"] == 1, "+0.25", "Ends in negative balance", "-0.25", "Balance is sufficient"),
        "Short Duration": (user["transaction_duration"] <= 2, "+0.25", "Suspiciously fast transaction", "-0.25", "Normal duration"),
        "Young Age": (user["customer_age"] < 24, "+0.25", "Very young customer", "-0.25", "Customer age is mature")
    }

    heatmap_data = []
    annotations = []
    summary_lines = []

    for factor, weight, condition, pos_desc, neg_desc in score_factors:
        impact = weight if condition else -weight
        sign = "+" if impact > 0 else "–"
        desc = pos_desc if condition else neg_desc
        status = "🔴 High Risk" if impact > 0 else "🔵 Low Risk"

        heatmap_data.append((factor, impact, f"{sign}{abs(weight)}"))
        annotations.append(desc)
        summary_lines.append(f"- **{factor}** → {desc} → **{sign}{abs(weight)}** ({status})")

    heatmap_df = pd.DataFrame(heatmap_data, columns=["Feature", "RiskScore", "Label"]).set_index("Feature")
    heatmap_df["Explanation"] = annotations
    heatmap_df = heatmap_df.reindex(heatmap_df['RiskScore'].abs().sort_values(ascending=False).index)  # Sort biggest impact on top

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_df[["RiskScore"]],
        annot=heatmap_df[["Label"]],
        fmt="",
        cmap="RdBu_r",   # Red-Blue reversed
        center=0,        # Center at 0
        linewidths=0.5,
        cbar_kws={"label": "Fraud Likelihood Score"},
        ax=ax
    )
    plt.title("Anomaly Heatmap", fontsize=14)
    st.pyplot(fig)

    st.markdown("<h3 style='font-family: Arial;'>📋 Heatmap Summary Explanation:</h3>", unsafe_allow_html=True)
    for line in summary_lines:
        st.markdown(line)

    if st.button("🔄 Reset Email Sent Flag (Dev Only)"):
        st.session_state.email_sent = False
        st.success("✅ Email sent flag has been reset. Email button will now reappear.")

    if (
        d['result'] == "Fraudulent"
        and d['confidence_score'] >= 50
        and d.get('email') is not None
        and d.get('email').strip() != ""
        and not st.session_state.email_sent
    ):
        st.markdown("<h3 style='font-family: Arial;'>📧 Email Alert</h3>", unsafe_allow_html=True)
        st.markdown(f"**Ready to alert `{d['email']}` about the flagged transaction.**")

        if st.button("📧 Send Fraud Alert Email"):
            tx = "\n".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in d['user_input'].items()])
            email_sent = send_email_alert(
                to_email=d['email'].strip(),
                subject="🚨 FRAUD ALERT – Suspicious Transaction Detected",
                message=f"""A transaction was flagged with a **confidence level of {d['confidence_score']}%**.

Behavioral Risk Rating: {rating} / 5

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
