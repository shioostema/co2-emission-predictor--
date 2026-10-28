import streamlit as st
import joblib
import numpy as np
import pandas as pd
import lzma

st.set_page_config(page_title="Regional CO2 Emissions Predictor", layout="centered")

st.title("🌍 Regional CO2 Emissions Predictor")
st.write("This web application predicts average CO2 emissions based on SO2 and NOx hourly metrics.")

# Automatically decompress and load the model on the fly
@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load('scaler.pkl')
        # Open and decompress the LZMA compressed model file
        with lzma.open('co2_model.pkl.xz', 'rb') as f:
            model = joblib.load(f)
        return scaler, model
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

scaler, model = load_assets()

if scaler is None or model is None:
    st.warning("⚠️ Please make sure 'scaler.pkl' and 'co2_model.pkl.xz' are uploaded in the same GitHub folder as this app.py script.")
else:
    st.success("✅ Model & Scaler loaded successfully!")

    st.subheader("Input Environmental Metrics")
    so2_input = st.number_input("Average SO2 Emissions (lbs/MWh)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
    nox_input = st.number_input("Average NOx Emissions (lbs/MWh)", min_value=0.0, max_value=50.0, value=1.5, step=0.1)

    if st.button("Predict CO2 Average"):
        try:
            input_features = np.array([[so2_input, nox_input]])
            input_scaled = scaler.transform(input_features)
            prediction = model.predict(input_scaled)[0]

            st.markdown("--- ")
            st.metric(label="Predicted CO2 Average Emissions (lbs/MWh)", value=f"{prediction:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")