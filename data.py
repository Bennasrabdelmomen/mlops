import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Site and technology parameters
sites = ["Site_A", "Site_B", "Site_C", "Site_D"]
technologies = ["2G", "3G", "4G", "5G", "LTE", "WiFi"]

# Base values for users and RRB
base_users = {
    "2G": 70,
    "3G": 60,
    "4G": 50,
    "5G": 20,
    "LTE": 30,
    "WiFi": 40
}

base_rrb = {
    "2G": 100,
    "3G": 150,
    "4G": 200,
    "5G": 300,
    "LTE": 150,
    "WiFi": 100
}

base_temperature = {
    "Site_A": 25,
    "Site_B": 30,
    "Site_C": 20,
    "Site_D": 35
}

technology_temperature_influence = {
    "2G": 0.8,
    "3G": 1.0,
    "4G": 1.2,
    "5G": 1.5,
    "LTE": 1.1,
    "WiFi": 1.0
}

def generate_users(technology):
    mean = base_users[technology]
    std_dev = mean * 0.1
    return max(0, int(np.random.normal(mean, std_dev)))

def generate_rrb(technology):
    mean = base_rrb[technology]
    std_dev = mean * 0.15
    return max(0, int(np.random.normal(mean, std_dev)))

def generate_temperature(site, technology, users, rrb):
    base_temp = base_temperature[site]
    tech_influence = technology_temperature_influence[technology]
    temp = base_temp + tech_influence * (users * 0.05 + rrb * 0.02)
    temp_noise = np.random.normal(0, 2)
    return round(temp + temp_noise, 2)

def generate_data_for_timestamp(current_time):
    data = []
    for site in sites:
        for tech in technologies:
            users = generate_users(tech)
            rrb = generate_rrb(tech)
            temp = generate_temperature(site, tech, users, rrb)
            data.append({
                "timestamp": current_time,
                "site_name": site,
                "technology": tech,
                "users": users,
                "RRB": rrb,
                "temperature": temp
            })
    return data

def generate_data(start_date, end_date, interval_seconds=600, output_file="synthetic_data.csv"):
    """Generates synthetic data within a date range and saves it to a CSV file."""
    current_time = start_date
    end_time = end_date
    all_data = []

    print(f"Generating data from {start_date} to {end_date}...")

    while current_time <= end_time:
        data = generate_data_for_timestamp(current_time)
        all_data.extend(data)

        # Save data in chunks to avoid memory issues
        if len(all_data) >= 100000:
            df = pd.DataFrame(all_data)
            if not os.path.exists(output_file):
                df.to_csv(output_file, mode="w", header=True, index=False)
            else:
                df.to_csv(output_file, mode="a", header=False, index=False)
            all_data = []

        current_time += timedelta(seconds=interval_seconds)

    if all_data:
        df = pd.DataFrame(all_data)
        if not os.path.exists(output_file):
            df.to_csv(output_file, mode="w", header=True, index=False)
        else:
            df.to_csv(output_file, mode="a", header=False, index=False)

    print("Data generation complete.")
