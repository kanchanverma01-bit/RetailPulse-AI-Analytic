import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Page Configuration (Real apps always have this)
st.set_page_config(
    page_title="RetailPulse AI-Analytic",
    page_icon="🛍️",
    layout="wide"
)

# 2. Mock Data Generation (For realistic feel until data-engineering team gives real data)
@st.cache_data
def load_mock_data():
    dates = pd.date_range(start="2026-01-01", periods=100)
    data = pd.DataFrame({
        'Date': dates,
        'Sales': np.random.randint(20000, 50000, size=100),
        'Orders': np.random.randint(150, 500, size=100),
        'Customer_Satisfaction': np.random.uniform(4.0, 4.9, size=100).round(2),
        'Category': np.random.choice(['Electronics', 'Clothing', 'Home Decor', 'Groceries'], size=100)
    })
    return data

df = load_mock_data()

# 3. Sidebar Filters
st.sidebar.title("🛍️ RetailPulse Admin")
st.sidebar.subheader("Filter Insights")
selected_category = st.sidebar.multiselect(
    "Select Category", 
    options=df['Category'].unique(), 
    default=df['Category'].unique()
)

# Filter data based on sidebar selection
filtered_df = df[df['Category'].isin(selected_category)]

# 4. Main Header
st.title("📊 RetailPulse AI-Analytic Dashboard")
st.markdown("Real-time Customer Analytics & AI-Driven Demand Forecasting Platform.")
st.markdown("---")

# 5. KPI Metrics Row (High-level business numbers)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Revenue", value=f"${filtered_df['Sales'].sum():,}")
with col2:
    st.metric(label="Total Orders", value=f"{filtered_df['Orders'].sum():,}")
with col3:
    st.metric(label="Avg Customer Rating", value=f"⭐ {filtered_df['Customer_Satisfaction'].mean():.2f}")
with col4:
    st.metric(label="Forecasted Growth (Next Month)", value="+14.2%", delta="AI Predicted")

st.markdown("---")

# 6. Interactive Tabs for Detailed Analysis
tab1, tab2, tab3 = st.tabs(["📈 Sales Trends", "🎯 Customer Analytics", "🔮 Demand Forecasting"])

with tab1:
    st.subheader("Sales & Revenue Analytics Over Time")
    fig_sales = px.line(filtered_df, x='Date', y='Sales', title="Daily Revenue Breakdown", markers=True)
    st.plotly_chart(fig_sales, use_container_width=True)

with tab2:
    st.subheader("Customer Segment & Category Performance")
    fig_pie = px.pie(filtered_df, values='Sales', names='Category', title='Revenue Contribution by Category', hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    st.subheader("🤖 AI Demand Forecasting")
    st.info("Once Suraj (forecasting-model) pushes the ML models, the live future predictions will render here.")
    # Dummy future projection line
    future_dates = pd.date_range(start="2026-04-11", periods=10)
    future_pred = np.random.randint(45000, 60000, size=10)
    fig_forecast = px.bar(x=future_dates, y=future_pred, title="AI Predicted Sales for Next 10 Days", labels={'x': 'Future Date', 'y': 'Predicted Revenue'})
    st.plotly_chart(fig_forecast, use_container_width=True)
