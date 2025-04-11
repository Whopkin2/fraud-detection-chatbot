import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import xgboost as xgb
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
df = data.copy()
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df.drop("Fraud", axis=1)
y = df["Fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(probability=True)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
isolation_model = IsolationForest(contamination=0.05, random_state=42)

rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Evaluation Function
def evaluate_models():
    models = {
        "Random Forest": rf_model,
        "SVM": svm_model,
        "XGBoost": xgb_model
    }
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            "AUC-ROC": roc_auc_score(y_test, y_proba),
            "FPR": confusion_matrix(y_test, y_pred)[0, 1] / (confusion_matrix(y_test, y_pred)[0, 1] + confusion_matrix(y_test, y_pred)[0, 0]),
            "FNR": confusion_matrix(y_test, y_pred)[1, 0] / (confusion_matrix(y_test, y_pred)[1, 0] + confusion_matrix(y_test, y_pred)[1, 1])
        }
    return results

def predict_fraud(user_input):
    input_df = pd.DataFrame([user_input])
    for col in categorical_cols:
        if col in input_df:
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
    prediction = rf_model.predict(input_df)[0]
    probability = rf_model.predict_proba(input_df)[0][1]
    return prediction, round(probability * 100, 2)

questions = [
    "Transaction ID:",
    "Transaction Amount:",
    "Transaction Type:",
    "Origin Account Age:",
    "Destination Account Age:",
    "Customer Location:",
    "Time of Transaction:",
    "Previous Fraud History (Yes/No):"
]

keys = [
    "TransactionID", "Amount", "TransactionType", "OriginAccountAge",
    "DestinationAccountAge", "Location", "TransactionTime", "FraudHistory"
]

if "question_index" not in st.session_state:
    st.session_state.question_index = 0
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

st.markdown("## 🕵️ Fraud Detection Chatbot")
st.markdown("Answer a few questions and get a fraud risk assessment:")

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

        st.markdown("---")
        st.markdown("### 📊 Model Performance Summary")
        eval_results = evaluate_models()
        for model_name, metrics in eval_results.items():
            st.markdown(f"**{model_name}**")
            for metric, value in metrics.items():
                st.markdown(f"- {metric}: {round(value, 4)}")

