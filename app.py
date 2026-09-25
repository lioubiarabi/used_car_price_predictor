import streamlit as st
import pandas as pd
import joblib
from src.data_loader.cleaned_results import load_cleaned_data

# 1. Page Configuration
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")


# 2. Load Model and Data Structure (Cached so it doesn't reload on every click)
@st.cache_resource
def load_assets():
    model = joblib.load("data/models/xgboost_car_price_model.pkl")  # Adjust path if needed
    df = load_cleaned_data(file_path="data/processed/cleaned_results.csv")
    expected_columns = df.drop(columns=["selling_price"]).columns

    # Extract unique brand_models dynamically from the dummy columns
    brand_models = [col.replace('brand_model_', '') for col in expected_columns if col.startswith('brand_model_')]

    return model, expected_columns, sorted(brand_models)


model, expected_columns, brand_models = load_assets()

# 3. Web App Header
st.title("🚗 Used Car Price Predictor")
st.write("Adjust the details below to get an instant, AI-powered price estimate.")

# 4. Input Layout (Divided into two columns)
col1, col2 = st.columns(2)

with col1:
    year = st.slider("Manufacturing Year", min_value=1990, max_value=2024, value=2018)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=45000, step=5000)

    owner_map = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above': 4}
    owner_text = st.selectbox("Owner History", list(owner_map.keys()))
    owner = owner_map[owner_text]

with col2:
    brand_model = st.selectbox("Brand & Model", brand_models)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    seller = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission_text = st.radio("Transmission", ["Manual", "Automatic"])
    transmission = 1 if transmission_text == "Automatic" else 0

# 5. Prediction Logic
if st.button("Calculate Estimated Price", type="primary", use_container_width=True):

    # Create a blank canvas of zeros based on the exact columns the model expects
    user_car = pd.DataFrame(0, index=[0], columns=expected_columns)

    # Inject numerical and label-encoded inputs
    user_car.at[0, 'year'] = year
    user_car.at[0, 'km_driven'] = km_driven
    user_car.at[0, 'transmission'] = transmission
    user_car.at[0, 'owner'] = owner

    # Inject one-hot encoded inputs safely
    if f'fuel_{fuel}' in expected_columns:
        user_car.at[0, f'fuel_{fuel}'] = 1
    if f'seller_type_{seller}' in expected_columns:
        user_car.at[0, f'seller_type_{seller}'] = 1
    if f'brand_model_{brand_model}' in expected_columns:
        user_car.at[0, f'brand_model_{brand_model}'] = 1

    # Run the model
    with st.spinner("Calculating..."):
        prediction = model.predict(user_car)[0]

    # Display the result
    st.success(f"### 💰 Estimated Value: ₹{prediction:,.2f}")