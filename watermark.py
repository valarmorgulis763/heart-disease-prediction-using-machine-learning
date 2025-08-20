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
# Load model & metadata
# -----------------------------
model = pickle.load(open("heart_disease_model.sav", "rb"))
x_train = pickle.load(open("x_train_columns.pkl", "rb"))   # training columns
lambdas = pickle.load(open("lambdas.pkl", "rb"))           # Box-Cox lambda values

continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# -----------------------------
# PDF Generator
# -----------------------------
def generate_pdf(name, age, sex, probability, prediction_text, advice_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(160, 750, "Heart Disease Prediction Report")

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
    text_obj.textLines("Doctor’s Recommendations:\n" + advice_text)
    c.drawText(text_obj)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

st.sidebar.title("🏥 Medical Dashboard")
section = st.sidebar.radio("Navigate", ["Patient Info", "Medical Inputs", "Results", "Report"])

st.markdown(
    """
    <style>
    body {background-color: #f4f9ff;}
    .stMetric {background-color: #ffffff; border-radius: 10px; padding: 10px;}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Step 1: Patient Info
# -----------------------------
if section == "Patient Info":
    st.title("🩺 Patient Information")
    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    sex = st.radio("Gender", ["Male", "Female"])

# -----------------------------
# Step 2: Medical Inputs
# -----------------------------
elif section == "Medical Inputs":
    st.title("📊 Medical Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, step=1)
    with col2:
        chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, step=1)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    with col3:
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
        thalach = st.number_input("Maximum Heart Rate (thalach)", min_value=60, max_value=250, step=1)

    col4, col5, col6 = st.columns(3)
    with col4:
        exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    with col5:
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1)
    with col6:
        slope = st.selectbox("Slope (slope)", [0, 1, 2])

    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (thal)", [0, 1, 2, 3])

# -----------------------------
# Step 3: Prediction Results
# -----------------------------
elif section == "Results":
    st.title("📑 Prediction Results")

    if st.button("🔍 Run Prediction"):
        # Encode gender
        sex_val = 1 if sex == "Male" else 0

        # Prepare input data
        input_data = (age, sex_val, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal)
        original_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

        input_df = pd.DataFrame([input_data], columns=original_columns)
        input_df_encoded = pd.get_dummies(input_df, columns=['cp', 'restecg', 'thal'], drop_first=True)

        for feature in ['sex', 'fbs', 'exang', 'slope', 'ca']:
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

        # Result Text
        if prediction[0] == 0:
            prediction_text = "✅ No Heart Disease Detected"
            advice_text = "- Maintain a healthy diet\n- Regular exercise\n- Avoid smoking & alcohol\n- Routine check-ups"
        else:
            prediction_text = "⚠️ High Risk of Heart Disease"
            advice_text = "- Immediate cardiologist consultation\n- Heart-healthy diet\n- Reduce salt/sugar\n- Light doctor-approved exercise"

        # Display results
        col1, col2 = st.columns(2)
        col1.metric("Prediction", prediction_text)
        col2.metric("Risk Probability", f"{probability:.2f} %")

        st.subheader("Doctor’s Recommendations")
        st.info(advice_text)

        # Visualization
        st.subheader("📊 Health Parameters Overview")
        fig, ax = plt.subplots()
        values = [trestbps, chol, thalach, oldpeak]
        labels = ["Blood Pressure", "Cholesterol", "Max HR", "ST Depression"]
        ax.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_ylabel("Values")
        ax.set_title("Patient Medical Parameters")
        st.pyplot(fig)

# -----------------------------
# Step 4: Report Download
# -----------------------------
elif section == "Report":
    st.title("📄 Generate Medical Report")
    pdf_buffer = generate_pdf(name, age, sex, probability, prediction_text, advice_text)
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_buffer,
        file_name=f"Heart_Disease_Report_{name.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )
