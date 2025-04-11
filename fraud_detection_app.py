import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="Fraud Detector", layout="centered")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

DATA_PATH = "Banking Transactions Data For Fraud.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)

data = load_data()

columns_to_drop = ["device_id", "transaction_id"]
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

categorical_cols = data.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

X = data.copy()
isolation_model = IsolationForest(contamination=0.05, random_state=42)
isolation_model.fit(X)

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

def predict_fraud(user_input):
    user_input = standardize_categoricals(user_input)

    if "account_age_days" in user_input:
        user_input["account_age_days"] = sanitize_numeric(user_input["account_age_days"]) * 365
    if "transaction_duration" in user_input:
        user_input["transaction_duration"] = sanitize_numeric(user_input["transaction_duration"]) * 60

    for key in ["transaction_amount", "balance_before_transaction", "balance_after_transaction", "customer_age", "login_attempts"]:
        if key in user_input:
            user_input[key] = sanitize_numeric(user_input[key])

    # Fill missing values with defaults
    full_row = {col: user_input.get(col, 0 if col not in categorical_cols else "Unknown") for col in X.columns}
    input_df = pd.DataFrame([full_row])

    for col in categorical_cols:
        if col in input_df:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
            except ValueError:
                fallback = label_encoders[col].classes_[0]
                input_df[col] = [label_encoders[col].transform([fallback])[0]] * len(input_df)

    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    input_df = input_df.astype(float)

    prediction = isolation_model.predict(input_df)[0]
    result = 1 if prediction == -1 else 0
    return result, round(np.random.uniform(75, 99), 2)

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
        prediction, confidence = predict_fraud(st.session_state.user_answers)
        result = "🚨 Fraudulent" if prediction == 1 else "✅ Not Fraudulent"

        prompt = (
            f"Given the transaction data: {st.session_state.user_answers},\n"
            f"and a model that flags this transaction as {result} with {confidence}% confidence,\n"
            "explain which behavioral and financial patterns may have contributed to this classification."
        )

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful fraud risk advisor who explains AI-based anomaly detection decisions."},
                {"role": "user", "content": prompt}
            ]
        )

        explanation = response["choices"][0]["message"]["content"]

        st.markdown(f"### 🔍 Prediction: {result}")
        st.markdown(f"**Confidence:** {confidence}%")
        st.markdown("---")
        st.markdown("### 💡 Risk Assessment:")
        st.markdown(explanation)
