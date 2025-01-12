from airflow import airflow scheduler
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle
from prophet import Prophet

# Import your existing data generation function
from data import generate_data

# Paths for data and model
DATA_FILE = "synthetic_data.csv"
MODEL_FILE = "prophet_forecasting_model.pkl"
OUTPUT_FORECAST_FILE = "forecasted_temperature_new.csv"


def generate_new_data():
    """Generate new data using the existing data generation function."""
    start_date = datetime.now()
    end_date = start_date + timedelta(days=1)  # Generate data for one day
    generate_data(start_date, end_date, output_file=DATA_FILE)
    print("New data generation completed.")


def preprocess_data():
    """Preprocess data for Prophet model."""
    data = pd.read_csv(DATA_FILE)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["timestamp"] = data["timestamp"].dt.floor("H")

    # Aggregate data
    technology_columns = [col for col in data.columns if col.startswith("tech_")]
    aggregated_data = data.groupby(["timestamp", "site_name"]).agg(
        {
            "temperature": "mean",
            "RRB": "mean",
            "users": "sum",
            **{col: "sum" for col in technology_columns}
        }
    ).reset_index()

    # Prepare for Prophet
    aggregated_data = aggregated_data.rename(columns={"timestamp": "ds", "temperature": "y"})
    site_columns = pd.get_dummies(aggregated_data["site_name"], prefix="site").astype(int)
    aggregated_data = pd.concat([aggregated_data, site_columns], axis=1)
    aggregated_data.drop(columns=["site_name"], inplace=True)
    aggregated_data.to_csv(DATA_FILE, index=False)
    print("Preprocessing completed.")


def forecast_with_new_data():
    """Use the pre-trained Prophet model for forecasting."""
    # Load the model
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)

    # Load preprocessed data
    data = pd.read_csv(DATA_FILE)

    # Create future dataframe for prediction
    future = model.make_future_dataframe(periods=24, freq="H")  # 24 hours forecast
    technology_columns = [col for col in data.columns if col.startswith("tech_")]
    site_columns = [col for col in data.columns if col.startswith("site_")]
    future = future.merge(
        data[["ds", "RRB", "users"] + technology_columns + site_columns],
        on="ds",
        how="left"
    )

    # Fill missing values
    future["RRB"].fillna(data["RRB"].mean(), inplace=True)
    future["users"].fillna(0, inplace=True)
    for tech_col in technology_columns:
        future[tech_col].fillna(0, inplace=True)
    for site_col in site_columns:
        future[site_col].fillna(0, inplace=True)

    # Forecast
    forecast = model.predict(future)
    forecast.to_csv(OUTPUT_FORECAST_FILE, index=False)
    print("Forecasting completed.")


# Default arguments for Airflow DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
with DAG(
        "forecast_data_ingestion",
        default_args=default_args,
        description="Generate data, preprocess, and forecast using Prophet",
        schedule_interval=timedelta(days=1),
        start_date=datetime(2025, 1, 1),
        catchup=False,
) as dag:
    task_generate_data = PythonOperator(
        task_id="generate_new_data",
        python_callable=generate_new_data,
    )

    task_preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    task_forecast = PythonOperator(
        task_id="forecast_with_new_data",
        python_callable=forecast_with_new_data,
    )

    # Task dependencies
    task_generate_data >> task_preprocess_data >> task_forecast
