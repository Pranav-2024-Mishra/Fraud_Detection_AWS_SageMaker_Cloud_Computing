import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ✅ Load trained model
model = joblib.load("fraud_model.pkl")

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ================================
# ✅ HEADER
# ================================
st.title("💳 Fraud Detection Dashboard")
st.markdown(
    "This dashboard allows *real-time fraud prediction* for single and batch transactions. "
    "Powered by AWS SageMaker (Training) + Streamlit Cloud (Deployment)."
)

# ================================
# ✅ SIDEBAR NAVIGATION
# ================================
menu = ["Single Prediction", "Batch Prediction (CSV Upload)", "Analytics Dashboard"]
choice = st.sidebar.selectbox("🔍 Navigate", menu)

# ===========================================
#  1️. SINGLE PREDICTION PAGE
# ===========================================
if choice == "Single Prediction":
    st.header("🔍 Single Transaction Fraud Prediction")

    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    inputs = []
    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):
        if i % 2 == 0:
            value = col1.number_input(f"{feature}", value=0.0)
        else:
            value = col2.number_input(f"{feature}", value=0.0)
        inputs.append(value)

    if st.button("Predict Fraud"):
        data = np.array(inputs).reshape(1, -1)
        prediction = model.predict(data)[0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error("🚨 *Fraud Detected!*")
        else:
            st.success(" *Transaction is Legit*")

# ===========================================
#  2️. BATCH PREDICTION SECTION
# ===========================================
elif choice == "Batch Prediction (CSV Upload)":
    st.header("📂 Batch Prediction for Multiple Transactions")

    uploaded_file = st.file_uploader("Upload a CSV file with transaction features", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(" Uploaded Data Preview:")
        st.dataframe(df.head())

        # Ensure model compatibility
        predictions = model.predict(df)
        df["Prediction"] = predictions

        st.write(" Output with Predictions:")
        st.dataframe(df.head())

        # Download output file
        csv_download = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_download,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

# ===========================================
# ✅ 3️.  ANALYTICS DASHBOARD
# ===========================================
elif choice == "Analytics Dashboard":
    st.header("📊 Fraud Analytics Dashboard")

    st.write(
        "Upload your dataset with *Prediction* column to view analytics."
    )

    file = st.file_uploader("Upload predictions CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        # Pie Chart – Fraud vs Legit
        st.subheader("Fraud vs Legit Distribution")
        pie_data = df["Prediction"].value_counts().reset_index()
        pie_data.columns = ["Class", "Count"]

        pie_chart = px.pie(
            pie_data,
            values="Count",
            names="Class",
            color="Class",
            title="Fraud vs Legit Transactions"
        )
        st.plotly_chart(pie_chart)

        # Amount Distribution
        if "Amount" in df.columns:
            st.subheader("Transaction Amount Distribution")
            fig_amount = px.histogram(
                df,
                x="Amount",
                color="Prediction",
                title="Amount Distribution by Class",
                nbins=50
            )
            st.plotly_chart(fig_amount)

        # Time Trend (if Time exists)
        if "Time" in df.columns:
            st.subheader("Fraud Over Time")
            fig_time = px.scatter(
                df,
                x="Time",
                y="Prediction",
                title="Fraud Cases Over Time",
                color="Prediction"
            )
            st.plotly_chart(fig_time)

# ================================
# * FOOTER
# ================================
st.write("---")
st.caption(
    "Developed by Rajan Kumar, Mansha Pandey & Pranav Mishra | "
    "Cloud Computing Project | AWS SageMaker + Streamlit Cloud"
)
