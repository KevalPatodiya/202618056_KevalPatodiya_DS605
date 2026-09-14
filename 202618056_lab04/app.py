import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Load the saved pipeline (preprocessing + trained model)
# ---------------------------------------------------------
@st.cache_resource

def cap_minimum_nights(df_num):
    df_copy = df_num.copy()
    df_copy['minimum_nights'] = df_copy['minimum_nights'].clip(upper=30)
    return df_copy
def load_model():
    return joblib.load("airbnb_price_model.pkl")

model = load_model()

st.set_page_config(page_title="NYC Airbnb Price Predictor", page_icon="🏙️")
st.title("🏙️ NYC Airbnb Price Predictor")
st.write(
    "Enter listing details below to get an estimated nightly price, "
    "based on a Random Forest model trained on the NYC Airbnb Open Data dataset."
)

st.divider()

# ---------------------------------------------------------
# Input fields — these match the exact columns the pipeline
# was trained on (see feature groups in the notebook):
#   text_col      = 'name'
#   recency_col    = ['days_since_last_review']
#   num_cols       = ['latitude', 'longitude', 'minimum_nights',
#                      'number_of_reviews', 'reviews_per_month',
#                      'calculated_host_listings_count', 'availability_365']
#   cat_cols       = ['neighbourhood_group', 'room_type']
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Listing name/title", "Cozy studio near the park")

    neighbourhood_group = st.selectbox(
        "Borough",
        ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
    )

    room_type = st.selectbox(
        "Room type",
        ["Entire home/apt", "Private room", "Shared room"],
    )

    latitude = st.number_input(
        "Latitude", value=40.7128, format="%.5f",
        help="NYC listings range roughly from 40.50 to 40.91",
    )

    longitude = st.number_input(
        "Longitude", value=-73.9971, format="%.5f",
        help="NYC listings range roughly from -74.24 to -73.71",
    )

with col2:
    minimum_nights = st.number_input("Minimum nights", min_value=1, value=2)

    number_of_reviews = st.number_input(
        "Number of reviews", min_value=0, value=10
    )

    reviews_per_month = st.number_input(
        "Reviews per month", min_value=0.0, value=1.0, step=0.1
    )

    calculated_host_listings_count = st.number_input(
        "Host's total listings", min_value=1, value=1
    )

    availability_365 = st.number_input(
        "Availability (days/year)", min_value=0, max_value=365, value=180
    )

st.divider()

has_reviews = st.checkbox("This listing has at least one review", value=True)
days_since_last_review = None
if has_reviews:
    days_since_last_review = st.number_input(
        "Days since last review", min_value=0, value=30
    )
else:
    st.caption("No reviews yet — the model will treat recency as missing.")

st.divider()

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
if st.button("Predict Price", type="primary"):
    input_df = pd.DataFrame(
    [
        {
            "name": name,
            "neighbourhood_group": neighbourhood_group,
            "room_type": room_type,
            "latitude": latitude,
            "longitude": longitude,
            "minimum_nights": minimum_nights,
            "number_of_reviews": number_of_reviews,
            "reviews_per_month": reviews_per_month,
            "calculated_host_listings_count": calculated_host_listings_count,
            "availability_365": availability_365,
            "days_since_last_review": (
                float(days_since_last_review) if days_since_last_review is not None else np.nan
            ),
        }
    ]
)

    # Model was trained on log1p(price), so invert with expm1
    log_pred = model.predict(input_df)
    price = np.expm1(log_pred)[0]

    st.success(f"### Estimated price: ${price:,.2f} per night")
    st.caption(
        "This is an estimate based on historical NYC listing data (2019) — "
        "actual prices depend on many factors not captured here (amenities, "
        "photos, seasonality, exact address, etc.)."
    )

st.divider()
st.caption(
    "Model: Random Forest Regressor · Trained on Kaggle NYC Airbnb Open Data (AB_NYC_2019.csv)"
)
