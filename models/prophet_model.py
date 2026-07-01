import pandas as pd

# # Load dataset
# df = pd.read_csv("data/cleaned_data/RetailPulse_Transactions_Member2.csv")

# # Show first 5 rows
# print(df.head())

# # Dataset information
# print(df.info())


# Load dataset
df = pd.read_csv("data/cleaned_data/RetailPulse_Transactions_Member2.csv")

# Convert OrderDate to datetime
df["OrderDate"] = pd.to_datetime(df["OrderDate"])

# Group by date and sum revenue
daily_sales = (
    df.groupby("OrderDate")["Revenue"]
      .sum()
      .reset_index()
)

print(daily_sales.head())
print(daily_sales.tail())

# Save daily sales
daily_sales.to_csv("reports/daily_sales.csv", index=False)

# print("Daily sales file saved successfully!")
from prophet import Prophet

# Prophet ke liye column names ds aur y hone chahiye
prophet_data = daily_sales.rename(
    columns={
        "OrderDate": "ds",
        "Revenue": "y"
    }
)

# Model create
model = Prophet()

# Model train
model.fit(prophet_data)

# print("Model trained successfully!")
# Create future dates (next 30 days)
future = model.make_future_dataframe(periods=30)

# Predict future sales
forecast = model.predict(future)

# Save prediction
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
    "reports/prediction.csv",
    index=False
)

print("Prediction file saved successfully!")