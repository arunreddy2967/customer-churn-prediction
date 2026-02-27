import streamlit as st
import pandas as pd
import pickle

# ---------------- TITLE ----------------
st.title("🚀 Customer Churn Prediction")
st.write(
    "This application predicts whether a customer will churn or not based on input details."
)

# ---------------- LOAD MODEL ----------------
try:
    # Load model
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    loaded_model = model_data["model"]
    feature_names = model_data["features_names"]

    # Load encoders
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()

# ---------------- INPUT FUNCTION ----------------
def user_input_features():
    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 1)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

    with col2:
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        MonthlyCharges = st.number_input("Monthly Charges", 0.0)
        TotalCharges = st.number_input("Total Charges", 0.0)

    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }

    return pd.DataFrame([data])


# ---------------- GET INPUT ----------------
input_data_df = user_input_features()

# ---------------- ENCODING ----------------
try:
    for column, encoder in encoders.items():
        if column in input_data_df.columns:
            input_data_df[column] = encoder.transform(input_data_df[column])
except Exception as e:
    st.error(f"Encoding error: {e}")
    st.stop()

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    try:
        # Arrange features
        input_data_df = input_data_df[feature_names]

        prediction = loaded_model.predict(input_data_df)
        prob = loaded_model.predict_proba(input_data_df)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("❌ Customer will Churn")
        else:
            st.success("✅ Customer will Stay")

        st.subheader("Prediction Probability")
        st.write(f"Churn: {prob[0][1]:.2f}")
        st.write(f"Stay: {prob[0][0]:.2f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")