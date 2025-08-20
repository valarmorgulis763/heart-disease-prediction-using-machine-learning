import pickle
import pandas as pd
import numpy as np
from scipy.stats import boxcox
import streamlit as st
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
import matplotlib.pyplot as plt

# -----------------------------
# Load model & training metadata
# -----------------------------
model = pickle.load(open("heart_disease_model.sav", "rb"))
x_train = pickle.load(open("x_train_columns.pkl", "rb"))
lambdas = pickle.load(open("lambdas.pkl", "rb"))

continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# -----------------------------
# PDF Generator
# -----------------------------
def generate_pdf(name, age, sex, probability, prediction_text, advice_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(150, 750, "Heart Disease Prediction Report")

    # Patient Info
    c.setFont("Helvetica", 12)
    c.drawString(50, 710, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 690, f"Patient Name: {name}")
    c.drawString(50, 670, f"Age: {age}")
    c.drawString(50, 650, f"Gender: {sex}")

    # Prediction
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, 620, f"Prediction: {prediction_text}")
    c.drawString(50, 600, f"Risk Probability: {probability:.2f}%")

    # Health Advice
    text_obj = c.beginText(50, 560)
    text_obj.setFont("Helvetica", 12)
    text_obj.textLines("Health Recommendations:\n" + advice_text)
    c.drawText(text_obj)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Sidebar
st.sidebar.title(" About This App")
st.sidebar.info(
    "This application predicts the risk of **Heart Disease** using a Machine Learning model.\n\n"
    "Enter your medical details, and the system will generate:\n"
    "- Risk Probability\n"
    "- Prediction Result\n"
    "- Personalized Health Advice\n"
    "- Downloadable PDF Medical Report"
)

# Main Title
st.markdown("<h1 style='text-align: center; color: #003366;'>🏥 Heart Disease Prediction Dashboard</h1>", unsafe_allow_html=True)

# Patient Info
st.markdown("### Patient Information")
col1, col2, col3 = st.columns(3)
with col1:
    name = st.text_input("Full Name")
with col2:
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
with col3:
    sex = st.radio("Gender", ["Male", "Female"])

# Medical Parameters
st.markdown("###  Medical Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, step=1)
    chol = st.number_input("Serum Cholesterol", min_value=100, max_value=600, step=1)
with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250, step=1)
with col3:
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

# Prediction
if st.button(" Predict", use_container_width=True):
    sex_val = 1 if sex == "Male" else 0
    input_data = (age, sex_val, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    original_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    input_df = pd.DataFrame([input_data], columns=original_columns)
    input_df_encoded = pd.get_dummies(input_df, columns=['cp', 'restecg', 'thal'], drop_first=True)
    features_to_convert = ['sex', 'fbs', 'exang', 'slope', 'ca']
    for feature in features_to_convert:
        input_df_encoded[feature] = input_df_encoded[feature].astype(int)

    missing_cols = set(x_train) - set(input_df_encoded.columns)
    for c in missing_cols:
        input_df_encoded[c] = 0
    input_df_encoded = input_df_encoded[x_train]

    input_df_encoded['oldpeak'] += 0.001
    for col in continuous_features:
        if col in lambdas:
            if (input_df_encoded[col] > 0).all():
                input_df_encoded[col] = boxcox(input_df_encoded[col], lmbda=lambdas[col])

    prediction = model.predict(input_df_encoded)
    probability = model.predict_proba(input_df_encoded)[0][1] * 100

    if prediction[0] == 0:
        prediction_text = " Low Risk: No Heart Disease Detected"
        advice_text = (
            "- Maintain a healthy diet \n"
            "- Exercise regularly \n"
            "- Avoid smoking  and alcohol \n"
            "- Regular health check-ups "
        )
        color = "green"
    else:
        prediction_text = "⚠️ High Risk: Potential Heart Disease"
        advice_text = (
            "- Consult a cardiologist immediately \n"
            "- Adopt a low-fat, heart-healthy diet \n"
            "- Reduce salt & sugar intake \n"
            "- Avoid smoking  & alcohol \n"
            "- Light exercise (as advised by doctor) 🚶‍♂️"
        )
        color = "red"

    # Display Result in Card
    st.markdown(
        f"""
        <div style='padding:20px; border-radius:10px; background-color:#f9f9f9;
        border-left:8px solid {color}; margin-bottom:20px;'>
        <h3 style='color:{color};'>{prediction_text}</h3>
        <p><b>Risk Probability:</b> {probability:.2f}%</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Show Advice
    st.markdown("### 🩺 Health Recommendations")
    st.write(advice_text)

    # Risk Pie Chart
    labels = ['Risk', 'Safe']
    values = [probability, 100 - probability]
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
           colors=['#ff4d4d', '#66b3ff'])
    ax.axis('equal')
    st.pyplot(fig)

    # PDF Download
    pdf_buffer = generate_pdf(name, age, sex, probability, prediction_text, advice_text)
    st.download_button(
        label="📄 Download Report",
        data=pdf_buffer,
        file_name=f"Heart_Disease_Report_{name.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )
