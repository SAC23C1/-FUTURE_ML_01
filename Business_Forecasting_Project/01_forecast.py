import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data (make sure file name matches exactly)
df = pd.read_csv("02_sales_data.csv")

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort data by date
df = df.sort_values('date')

# Create time index (0,1,2,...)
df['time'] = np.arange(len(df))

# Train Linear Regression model
model = LinearRegression()
model.fit(df[['time']], df['sales'])
from sklearn.metrics import mean_absolute_error

train_pred = model.predict(df[['time']])
error = mean_absolute_error(df['sales'], train_pred)

print("Model Error (MAE):", error)

# Predict next 3 months
future = pd.DataFrame({
    'time': np.arange(len(df), len(df)+3)
})
pred = model.predict(future)
print("Sales are showing an increasing trend.")
print("Future predicted sales:", pred)

# Generate future dates (Month End)
future_dates = pd.date_range(
    start=df['date'].iloc[-1], periods=4, freq='ME')[1:]

# Plot graph (Improved Visualization)
plt.figure()

# Actual data
plt.plot(df['date'], df['sales'], marker='o', label="Actual")

# Forecast data
plt.plot(future_dates, pred, marker='o', linestyle='--', label="Forecast")

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecasting")
plt.legend()

# Save output image
plt.savefig("forecast.png")

# Show graph
plt.show()
