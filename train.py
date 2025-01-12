import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

data_file = "synthetic_data.csv"
data = pd.read_csv(data_file)

data["timestamp"] = pd.to_datetime(data["timestamp"])

technology_columns = [col for col in data.columns if col.startswith("tech_")]
data[technology_columns] = data[technology_columns].astype(int)

data["timestamp"] = data["timestamp"].dt.floor("H")
aggregated_data = data.groupby(["timestamp", "site_name"]).agg(
    {
        "temperature": "mean",
        "RRB": "mean",
        "users": "sum",
        **{col: "sum" for col in technology_columns}
    }
).reset_index()

aggregated_data = aggregated_data.rename(columns={"timestamp": "ds", "temperature": "y"})

site_columns = pd.get_dummies(aggregated_data["site_name"], prefix="site").astype(int)
aggregated_data = pd.concat([aggregated_data, site_columns], axis=1)
aggregated_data.drop(columns=["site_name"], inplace=True)

train_data = aggregated_data[aggregated_data["ds"] < "2024-12-01"]
test_data = aggregated_data[aggregated_data["ds"] >= "2024-12-01"]

model = Prophet()
for tech_col in technology_columns:
    model.add_regressor(tech_col)
model.add_regressor("RRB")
model.add_regressor("users")
for site_col in site_columns.columns:
    model.add_regressor(site_col)

model.fit(train_data)

future = model.make_future_dataframe(periods=len(test_data), freq="H")  # Use hourly frequency

future = future.merge(
    aggregated_data[["ds", "RRB", "users"] + technology_columns + list(site_columns.columns)],
    on="ds",
    how="left"
)

future["RRB"].fillna(train_data["RRB"].mean(), inplace=True)
future["users"].fillna(0, inplace=True)
for tech_col in technology_columns:
    future[tech_col].fillna(0, inplace=True)
for site_col in site_columns.columns:
    future[site_col].fillna(0, inplace=True)

forecast = model.predict(future)

model.plot(forecast)
model.plot_components(forecast)

forecast_test = forecast[forecast["ds"].isin(test_data["ds"])]
y_true = test_data["y"].values
y_pred = forecast_test["yhat"].values

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

forecast.to_csv("forecasted_temperature_with_regressors_and_sites.csv", index=False)
import pickle

# Save the model
with open('prophet_forecasting_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully!")
