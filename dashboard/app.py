import os
import sys
import streamlit as st
import pandas as pd
import joblib

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.data_loader.cleaned_results import load_cleaned_data
from src.data_processing.clean import clean_data

# 1. Page Configuration
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# 2. Load Model and Data Structure (Cached so it doesn't reload on every click)
@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, "data", "models", "xgboost_car_price_model.pkl")
    cleaned_data_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_results.csv")
    raw_data_path = os.path.join(BASE_DIR, "data", "raw", "car-prices.csv")

    model = joblib.load(model_path)
    df = load_cleaned_data(cleaned_data_path)
    expected_columns = df.drop(columns=["selling_price"]).columns

    # Extract unique brand_models dynamically matching categorical encoding used in training
    raw_df = pd.read_csv(raw_data_path)
    c_df = clean_data(raw_df)
    c_df['brand_model'] = c_df['brand'] + ' ' + c_df['model']
    brand_models = sorted(c_df['brand_model'].unique().tolist())

    return model, expected_columns, brand_models

model, expected_columns, brand_models = load_assets()

# 3. Web App Header
st.title("🚗 Used Car Price Predictor")
st.write("Adjust the details below to get an instant, AI-powered price estimate.")

# 4. Input Layout (Divided into two columns)
col1, col2 = st.columns(2)

with col1:
    year = st.slider("Manufacturing Year", min_value=1990, max_value=2024, value=2018)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=45000, step=5000)

    owner_map = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above': 4,
        'Test Drive Car': 0
    }
    owner_text = st.selectbox("Owner History", list(owner_map.keys()))
    owner = owner_map[owner_text]

with col2:
    default_brand_idx = brand_models.index("Maruti Swift") if "Maruti Swift" in brand_models else 0
    brand_model = st.selectbox("Brand & Model", brand_models, index=default_brand_idx)

    fuel_map = {'Petrol': 1, 'Diesel': 0, 'CNG': 2, 'LPG': 3, 'Electric': 4}
    fuel = st.selectbox("Fuel Type", list(fuel_map.keys()))

    seller_map = {'Individual': 0, 'Dealer': 1, 'Trustmark Dealer': 2}
    seller = st.selectbox("Seller Type", list(seller_map.keys()))

    transmission_map = {'Manual': 0, 'Automatic': 1}
    transmission_text = st.radio("Transmission", list(transmission_map.keys()))
    transmission = transmission_map[transmission_text]

# 5. Prediction Logic
if st.button("Calculate Estimated Price", type="primary", use_container_width=True):

    # Map brand_model to its category code
    brand_model_code = brand_models.index(brand_model) if brand_model in brand_models else 0

    # Build input matching the exact features expected by the model
    user_car = pd.DataFrame([{
        'year': year,
        'km_driven': float(km_driven),
        'fuel': fuel_map[fuel],
        'seller_type': seller_map[seller],
        'transmission': transmission,
        'owner': owner,
        'brand_model': brand_model_code
    }])[expected_columns]

    # Run the model
    with st.spinner("Calculating..."):
        prediction = model.predict(user_car)[0]
        prediction = max(prediction, 15000.0)

    # Display the result
    st.success(f"### Estimated Value: {prediction:,.2f}")