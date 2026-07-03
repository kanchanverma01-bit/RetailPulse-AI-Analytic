# 🛍️ RetailPulse AI – Smart Retail Analytics & Demand Forecasting Platform

## 📌 Project Overview

RetailPulse AI is an end-to-end AI-powered retail analytics platform designed to help businesses make data-driven decisions through customer analytics, demand forecasting, and intelligent dashboards.

The platform combines Machine Learning, Customer Segmentation, Churn Prediction, Demand Forecasting, and Interactive Business Intelligence to provide actionable insights for retailers.

---

## 🚀 Key Features

### 📊 Interactive Dashboard
- Real-time business KPIs
- Sales & Revenue Analysis
- Customer Insights
- Interactive Visualizations

### 📈 Demand Forecasting
- Prophet-based forecasting
- Future sales prediction
- Trend & seasonality analysis

### 👥 Customer Segmentation
- RFM Analysis
- K-Means Clustering
- DBSCAN Clustering
- High-value customer identification

### ⚠️ Customer Churn Prediction
- XGBoost Classification Model
- Churn Probability Score
- Risk Categorization
- Customer Retention Insights

### 📉 Business Analytics
- Correlation Analysis
- Revenue Analysis
- Customer Behaviour Analysis
- Inventory Insights

---

# 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Forecasting | Prophet |
| Hyperparameter Tuning | Optuna |
| Visualization | Matplotlib, Plotly |
| Version Control | Git & GitHub |

---

# 📂 Project Structure

```
RetailPulse-AI-Analytic/
│
├── dashboard/
│   ├── app.py
│   └── README.md
│
├── data/
│   ├── raw_data/
│   ├── cleaned_data/
│   └── processed_data/
│
├── models/
│
├── notebook/
│
├── reports/
│
├── requirements.txt
└── README.md
```

---

# 📊 Machine Learning Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Customer Segmentation
6. Churn Prediction
7. Demand Forecasting
8. Dashboard Development
9. Business Insights

---

# ⚡ Challenges Faced

## Challenge 1

The original synthetic dataset contained randomly assigned churn labels, which showed very weak correlation with customer behaviour.

### Solution

A new behavioural churn definition was created using:

- Customer activity
- Purchase frequency
- Recency
- Monetary value
- Complaint history
- Inactivity period

This resulted in more realistic churn prediction.

---

## Challenge 2

Improving model performance.

### Solution

Optuna was used for hyperparameter optimization, improving the ROC-AUC score and enhancing prediction performance.

---

# 📈 Model Performance

| Model | Performance |
|--------|------------|
| Customer Churn Prediction | ROC-AUC ≈ 0.96 |
| Customer Segmentation | K-Means + DBSCAN |
| Demand Forecasting | Prophet |

---

# 💼 Business Impact

- Better customer retention
- Improved demand forecasting
- Data-driven decision making
- Inventory optimization
- High-value customer identification
- Actionable business insights

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/kanchanverma01-bit/RetailPulse-AI-Analytic.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the dashboard

```bash
streamlit run dashboard/app.py
```

---

# 📸 Dashboard Modules

- Home
- Sales Dashboard
- Demand Forecasting
- Customer Segmentation
- Churn Prediction
- Reports

---

# 👩‍💻 Contributors

- Kanchan Verma
- Neha Sonkar
- Sachin Rawat
- Suraj Yadav
---

# 📄 License

This project was developed for academic and internship purposes.

---

# ⭐ Thank You

Thank you for exploring **RetailPulse AI**.

*"Turning Retail Data into Smart Business Decisions."*
