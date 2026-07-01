import streamlit as st
import pandas as pd
import os
from sklearn.cluster import KMeans

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RetailPulse AI Dashboard", layout="wide")

st.title("📊 RetailPulse AI Powered Dashboard")

# ---------------- DATA LOAD ----------------
file_path = os.path.join("data", "processed_data")

@st.cache_data
def load_data():
    transactions = pd.read_excel(
        os.path.join(file_path, "RetailPulse_Transactions_Member2.xlsx")
    )
    customers = pd.read_excel(
        os.path.join(file_path, "RetailPulse_Customer_Final.xlsx")
    )
    return transactions, customers

df_transactions, df_customers = load_data()

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "📌 Modules",
    ["Home", "Demand Forecasting", "Customer Segmentation", "Churn Prediction", "Inventory Optimization"]
)

# ---------------- HOME ----------------
if page == "Home":
    st.subheader("Overview")
    st.dataframe(df_transactions.head())
    st.dataframe(df_customers.head())

# ---------------- DEMAND FORECASTING ----------------
elif page == "Demand Forecasting":
    st.subheader("📈 Demand Forecasting")

    numeric = df_transactions.select_dtypes(include='number')

    if numeric.shape[1] > 0:
        st.line_chart(numeric)
    else:
        st.warning("No numeric columns found")

# ---------------- CUSTOMER SEGMENTATION ----------------
elif page == "Customer Segmentation":
    st.subheader("👥 Customer Segmentation (KMeans)")

    data = df_customers.select_dtypes(include='number').dropna()

    if data.shape[1] >= 2:
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_customers["Cluster"] = kmeans.fit_predict(data)

        st.dataframe(df_customers.head())
        st.bar_chart(df_customers["Cluster"].value_counts())
    else:
        st.warning("Not enough numeric columns for clustering")

# ---------------- CHURN PREDICTION ----------------
elif page == "Churn Prediction":
    st.subheader("⚠️ Churn Prediction (Simple Score)")

    df_customers["Churn_Score"] = df_customers.select_dtypes(include='number').mean(axis=1)

    st.dataframe(df_customers[["Churn_Score"]].head())
    st.line_chart(df_customers["Churn_Score"])

# ---------------- INVENTORY OPTIMIZATION ----------------
elif page == "Inventory Optimization":
    st.subheader("📦 Inventory Optimization")

    numeric = df_transactions.select_dtypes(include='number')

    if numeric.shape[1] > 0:
        st.write("Average Demand per Product")
        st.bar_chart(numeric.mean())
    else:
        st.warning("No numeric data found")