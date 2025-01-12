import streamlit as st
import pandas as pd
import pickle
from prophet.plot import plot_plotly
from prophet.serialize import model_from_json


# Load the trained Prophet model
def load_model():
    with open('prophet_forecasting_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


# Streamlit app
st.title("Prophet Forecasting App")
st.markdown("This app allows you to use a pre-trained Prophet model for forecasting.")

# Load the model
model = load_model()
st.success("Model loaded successfully!")

# Upload future data or define forecast period
st.sidebar.header("Input Options")
input_type = st.sidebar.radio("Select input method:", ["Upload Data", "Specify Forecast Period"])

if input_type == "Upload Data":
    st.sidebar.markdown("**Upload your CSV file with the necessary regressors.**")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        future_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(future_data)
elif input_type == "Specify Forecast Period":
    periods = st.sidebar.number_input("Forecast Period (hours):", min_value=1, max_value=1000, value=24, step=1)
    freq = st.sidebar.selectbox("Frequency:", ["H", "D", "W", "M"], index=0)
    future_data = model.make_future_dataframe(periods=periods, freq=freq)
    st.write(f"Generated future DataFrame for {periods} {freq} periods.")
    st.write(future_data)

# Forecasting
if st.button("Generate Forecast"):
    if input_type == "Specify Forecast Period":
        st.info("Forecasting based on specified periods...")
    elif input_type == "Upload Data":
        st.info("Forecasting based on uploaded data...")

    # Fill missing regressors (if applicable)
    regressors = [col for col in future_data.columns if col not in ["ds"]]
    for reg in regressors:
        if reg not in model.extra_regressors:
            st.warning(f"Missing regressor: {reg}. Please include it.")
            st.stop()
        future_data[reg].fillna(0, inplace=True)

    # Make predictions
    forecast = model.predict(future_data)

    # Show results
    st.write("Forecasted Data:")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head())

    # Plot results
    st.write("Forecast Plot:")
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig)

    st.write("Forecast Components:")
    st.pyplot(model.plot_components(forecast))

