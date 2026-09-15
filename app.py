import streamlit as st
import pickle
import numpy as np

# -----------------------------------
# Load Machine Learning Models
# -----------------------------------

heart_model = pickle.load(open("heart_model.pkl", "rb"))
diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))
kidney_model = pickle.load(open("kidney_model.pkl", "rb"))
liver_model = pickle.load(open("liver_model.pkl", "rb"))


# -----------------------------------
# Page Title
# -----------------------------------

st.title("🏥 Healthcare Multiple Disease Risk Prediction")

st.write(
    "This application uses Machine Learning to predict the risk "
    "of multiple diseases based on patient health information."
)


# -----------------------------------
# Disease Selection
# -----------------------------------

disease = st.selectbox(
    "Select Disease",
    [
        "Heart Disease",
        "Diabetes",
        "Kidney Disease",
        "Liver Disease"
    ]
)


# ===================================
# HEART DISEASE
# ===================================

if disease == "Heart Disease":

    st.subheader("❤️ Heart Disease Risk Prediction")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=40
    )

    systolic_bp = st.number_input(
        "Systolic Blood Pressure",
        min_value=70,
        max_value=250,
        value=120
    )

    diastolic_bp = st.number_input(
        "Diastolic Blood Pressure",
        min_value=40,
        max_value=150,
        value=80
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=400,
        value=180
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=24.0
    )

    heart_rate = st.number_input(
        "Heart Rate",
        min_value=30,
        max_value=200,
        value=75
    )

    if st.button("Predict Heart Disease Risk"):

        input_data = np.array([[
            age,
            systolic_bp,
            diastolic_bp,
            cholesterol,
            bmi,
            heart_rate
        ]])

        prediction = heart_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Higher Risk of Heart Disease")
        else:
            st.success("✅ Lower Risk of Heart Disease")


# ===================================
# DIABETES
# ===================================

elif disease == "Diabetes":

    st.subheader("🩸 Diabetes Risk Prediction")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=40
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=50,
        max_value=300,
        value=100
    )

    hba1c = st.number_input(
        "HbA1c",
        min_value=3.0,
        max_value=15.0,
        value=5.5
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=24.0
    )

    systolic_bp = st.number_input(
        "Systolic Blood Pressure",
        min_value=70,
        max_value=250,
        value=120
    )

    if st.button("Predict Diabetes Risk"):

        input_data = np.array([[
            age,
            glucose,
            hba1c,
            bmi,
            systolic_bp
        ]])

        prediction = diabetes_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Higher Risk of Diabetes")
        else:
            st.success("✅ Lower Risk of Diabetes")


# ===================================
# KIDNEY DISEASE
# ===================================

elif disease == "Kidney Disease":

    st.subheader("🫘 Kidney Disease Risk Prediction")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=40
    )

    systolic_bp = st.number_input(
        "Systolic Blood Pressure",
        min_value=70,
        max_value=250,
        value=120
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=50,
        max_value=300,
        value=100
    )

    creatinine = st.number_input(
        "Creatinine",
        min_value=0.1,
        max_value=10.0,
        value=1.0
    )

    hemoglobin = st.number_input(
        "Hemoglobin",
        min_value=5.0,
        max_value=20.0,
        value=14.0
    )

    if st.button("Predict Kidney Disease Risk"):

        input_data = np.array([[
            age,
            systolic_bp,
            glucose,
            creatinine,
            hemoglobin
        ]])

        prediction = kidney_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Higher Risk of Kidney Disease")
        else:
            st.success("✅ Lower Risk of Kidney Disease")


# ===================================
# LIVER DISEASE
# ===================================

elif disease == "Liver Disease":

    st.subheader("🫀 Liver Disease Risk Prediction")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=40
    )

    bilirubin = st.number_input(
        "Bilirubin",
        min_value=0.1,
        max_value=20.0,
        value=1.0
    )

    alt = st.number_input(
        "ALT",
        min_value=1,
        max_value=500,
        value=30
    )

    ast = st.number_input(
        "AST",
        min_value=1,
        max_value=500,
        value=30
    )

    albumin = st.number_input(
        "Albumin",
        min_value=1.0,
        max_value=6.0,
        value=4.0
    )

    if st.button("Predict Liver Disease Risk"):

        input_data = np.array([[
            age,
            bilirubin,
            alt,
            ast,
            albumin
        ]])

        prediction = liver_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Higher Risk of Liver Disease")
        else:
            st.success("✅ Lower Risk of Liver Disease")


# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")

st.caption(
    "Educational project using synthetic healthcare data. "
    "This application is not intended for medical diagnosis."
)
