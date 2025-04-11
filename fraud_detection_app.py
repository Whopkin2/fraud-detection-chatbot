import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

st.set_page_config(page_title="Fraud Detector", layout="centered")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

DATA_PATH = "Banking Transactions Data For Fraud.xlsx"

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)
    return df

data = load_data()

# Preprocess
categorical_cols = data.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Assume the last column is the target (labeled fraud/no-fraud)
target_col = data.columns[-1]  # adjust manually if needed
y = data[target_col]
X = data.drop(columns=[target_col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict_fraud(user_input):
    input_df = pd.DataFrame([user_input])
    for col in categorical_cols:
        if col in input_df:
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return prediction, round(probability * 100, 2)

questions = [
    "Transaction Amount:",
    "Transaction Type:",
    "Account Age (Days):",
    "Is this International? (Yes/No):",
    "Time of Day:",
    "Customer Age:",
    "Branch Code:",
    "Device ID:",
    "Transaction Method:",
    "Balance Before Transaction:",
    "Balance After Transaction:",
    "Login Attempts:",
    "Transaction Duration (sec):"
]

keys = [
    "transaction_amount", "transaction_type", "account_age_days",
    "is_international", "time_of_day", "customer_age", "branch_code",
    "device_id", "transaction_method", "balance_before_transaction",
    "balance_after_transaction", "login_attempts", "transaction_duration"
]

if "question_index" not in st.session_state:
    st.session_state.question_index = 0
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

st.markdown("## 🕵️ Fraud Detection Chatbot")
st.markdown("Input values for a hypothetical transaction. The bot will assess fraud risk using past labeled data:")

for message in st.session_state.chat_log:
    st.markdown(message, unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    if st.session_state.question_index < len(questions):
        user_input = st.text_input(questions[st.session_state.question_index])
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
        prediction, probability = predict_fraud(st.session_state.user_answers)
        result = "🚨 Fraudulent" if prediction == 1 else "✅ Not Fraudulent"

        prompt = (
            f"Given the transaction data: {st.session_state.user_answers},\n"
            f"and a model that predicts this transaction as {result} with {probability}% confidence,\n"
            "summarize the risk factors that may have contributed to this decision."
        )

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful fraud risk advisor who explains AI-based fraud predictions."},
                {"role": "user", "content": prompt}
            ]
        )

        explanation = response["choices"][0]["message"]["content"]

        st.markdown(f"### 🔍 Prediction: {result}")
        st.markdown(f"**Confidence:** {probability}%")
        st.markdown("---")
        st.markdown("### 💡 Risk Assessment:")
        st.markdown(explanation)
