import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load Dataset
df = pd.read_csv("data/cleaned_data/RetailPulse_Transactions_Member2.csv")

print(df.head())

# Convert OrderDate to datetime
df["OrderDate"] = pd.to_datetime(df["OrderDate"])

# Create new features from date
df["Year"] = df["OrderDate"].dt.year
df["Month"] = df["OrderDate"].dt.month
df["Day"] = df["OrderDate"].dt.day
df["Weekday"] = df["OrderDate"].dt.dayofweek

print(df[["OrderDate", "Year", "Month", "Day", "Weekday"]].head())

# Encode categorical column
encoder = LabelEncoder()

df["PreferedOrderCat"] = encoder.fit_transform(df["PreferedOrderCat"])

print(df[["PreferedOrderCat"]].head())

# Features (Input)
X = df[
    [
        "PreferedOrderCat",
        "CashbackAmount",
        "CurrentStockLevel",
        "SafetyStock",
        "ReorderPoint",
        "LeadTimeDays",
        "Year",
        "Month",
        "Day",
        "Weekday",
    ]
]

# Target (Output)
y = df["Revenue"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Create XGBoost Model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

print("XGBoost Model Trained Successfully!")

y_pred = model.predict(X_test)
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

prediction_df = X_test.copy()

prediction_df["ActualRevenue"] = y_test
prediction_df["PredictedRevenue"] = y_pred

prediction_df.to_csv("reports/xgb_prediction.csv", index=False)

print("Prediction file saved successfully!")

import matplotlib.pyplot as plt

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print(importance)

plt.figure(figsize=(10,5))

plt.bar(importance["Feature"], importance["Importance"])

plt.xticks(rotation=45)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("XGBoost Feature Importance")

plt.tight_layout()

plt.savefig("reports/feature_importance.png")

#plt.show()

print("Feature importance graph saved successfully!")

last_date = df["OrderDate"].max()

print("Last Date:", last_date)

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

future_df = pd.DataFrame({
    "OrderDate": future_dates
})

future_df["Year"] = future_df["OrderDate"].dt.year
future_df["Month"] = future_df["OrderDate"].dt.month
future_df["Day"] = future_df["OrderDate"].dt.day
future_df["Weekday"] = future_df["OrderDate"].dt.dayofweek

print(future_df.head())

future_df["PreferedOrderCat"] = df["PreferedOrderCat"].mode()[0]

future_df["CashbackAmount"] = df["CashbackAmount"].mean()

future_df["LeadTimeDays"] = df["LeadTimeDays"].mean()

future_df["CurrentStockLevel"] = df["CurrentStockLevel"].mean()

future_df["SafetyStock"] = df["SafetyStock"].mean()

future_df["ReorderPoint"] = df["ReorderPoint"].mean()

print(df.columns.tolist())
print(future_df.columns.tolist())
print(future_df.head())

future_X = future_df[
    [
        "PreferedOrderCat",
        "CashbackAmount",
        "CurrentStockLevel",
        "SafetyStock",
        "ReorderPoint",
        "LeadTimeDays",
        "Year",
        "Month",
        "Day",
        "Weekday",
    ]
]

future_df["PredictedRevenue"] = model.predict(future_X)

future_df[["OrderDate", "PredictedRevenue"]].to_csv(
    "reports/xgb_future_prediction.csv",
    index=False
)

print("\n30 Days Future Revenue Prediction:")
print(future_df[["OrderDate", "PredictedRevenue"]].head(30))

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(
    future_df["OrderDate"],
    future_df["PredictedRevenue"],
    marker="o"
)
plt.title("30 Days Future Revenue Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Revenue")
plt.xticks(rotation=45)
plt.grid(True)

plt.savefig("reports/future_forecast.png")
plt.show()