# 🕵️ Fraud Detection Chatbot

This Streamlit app helps detect fraudulent banking transactions using multiple ML models and GPT explanations.

## 🚀 Setup Instructions

1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenAI key:
   ```env
   OPENAI_API_KEY=your-openai-api-key-here
   ```

4. Make sure the file `Banking Transactions Data For Fraud.xlsx` is in the same folder.
5. Run the app:
   ```bash
   streamlit run fraud_detection_app.py
   ```

## 🧠 Models Included
- Random Forest
- Support Vector Machine
- XGBoost
- Isolation Forest

## 📊 Metrics Evaluated
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
- False Positive Rate (FPR)
- False Negative Rate (FNR)
