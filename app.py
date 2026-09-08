"""
E-commerce Delivery Time Prediction — Streamlit app.

Loads the trained pipeline from model.pkl (preprocessing + LinearRegression,
trained on log(Delivery_Time_Hours)) and predicts delivery time in hours for
a single order entered through the UI. Does NOT retrain on startup.
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Delivery Time Predictor", page_icon="📦", layout="centered")

st.title("📦 E-commerce Delivery Time Predictor")
st.write(
    "Estimate how long a delivery will take based on order and logistics "
    "details. Built for ops/product teams to sanity-check expected delivery "
    "windows and flag orders that may need proactive customer communication."
)

# ---------------------------------------------------------------------------
# Load model (cached so it's loaded once, not on every interaction)
# ---------------------------------------------------------------------------
MODEL_PATH = "model.pkl"
DATA_PATH = "data.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_categories():
    df = pd.read_csv(DATA_PATH)
    categories = sorted(df["Product_Category"].dropna().unique().tolist())
    return categories


try:
    pipeline = load_model()
except FileNotFoundError:
    st.error(
        f"Could not find `{MODEL_PATH}`. Train the model first by running "
        f"`model_training.ipynb` end to end (it saves `model.pkl`), then "
        f"restart this app."
    )
    st.stop()

try:
    product_categories = load_categories()
except FileNotFoundError:
    st.warning(f"Could not find `{DATA_PATH}` — using a default category list.")
    product_categories = [
        "Apparel", "Books", "Electronics", "Health & Beauty",
        "Home & Kitchen", "Office Supplies", "Sports & Outdoors", "Toys & Games",
    ]

SHIPPING_MODES = ["Standard", "Express", "Economy", "Same Day"]

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.header("Order details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        product_category = st.selectbox("Product Category", product_categories)
        shipping_mode = st.selectbox("Shipping Mode", SHIPPING_MODES)
        order_value = st.number_input("Order Value ($)", min_value=0.0, value=100.0, step=1.0)
        package_weight = st.number_input("Package_Weight_Kg", min_value=0.0, value=2.0, step=0.1)
        warehouse_distance = st.number_input("Warehouse Distance Km", min_value=0.0, value=250.0, step=1.0)

    with col2:
        items_in_order = st.number_input("Items in Order", min_value=1, value=2, step=1, format="%d")
        processing_hours = st.number_input("Warehouse_Processing_Hours", min_value=0.0, value=6.0, step=0.5)
        courier_load = st.slider("Courier_Load_Index", min_value=0.5, max_value=1.5, value=1.0, step=0.01)
        traffic_index = st.slider("Traffic Index", min_value=0.5, max_value=2.0, value=1.2, step=0.01)

    submitted = st.form_submit_button("Predict delivery time")

# ---------------------------------------------------------------------------
# Validation + prediction
# ---------------------------------------------------------------------------
if submitted:
    errors = []
    if order_value < 0:
        errors.append("Order Value cannot be negative.")
    if package_weight < 0:
        errors.append("Package_Weight_Kg cannot be negative.")
    if warehouse_distance < 0:
        errors.append("Warehouse Distance Km cannot be negative.")
    if items_in_order < 1:
        errors.append("Items in Order must be at least 1.")
    if processing_hours < 0:
        errors.append("Warehouse_Processing_Hours cannot be negative.")
    if courier_load <= 0:
        errors.append("Courier_Load_Index must be positive.")
    if traffic_index <= 0:
        errors.append("Traffic Index must be positive.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        input_row = pd.DataFrame([{
            "Order Value": order_value,
            "Package_Weight_Kg": package_weight,
            "Warehouse Distance Km": warehouse_distance,
            "Items in Order": items_in_order,
            "Warehouse_Processing_Hours": processing_hours,
            "Courier_Load_Index": courier_load,
            "Traffic Index": traffic_index,
            "Product_Category": product_category,
            "Shipping_Mode": shipping_mode,
        }])

        # Model was trained on log(Delivery_Time_Hours); convert back to hours.
        pred_log_hours = pipeline.predict(input_row)[0]
        pred_hours = float(np.exp(pred_log_hours))

        st.success(f"**Estimated Delivery Time: {pred_hours:.1f} hours**")

        days = pred_hours / 24
        st.caption(f"≈ {days:.1f} days")

        st.subheader("What this means for an ops/product manager")
        st.write(
            f"- **Customer communication:** set the expected delivery window "
            f"around **{pred_hours:.0f} hours** (≈ {days:.1f} days) rather than "
            f"a generic default, reducing 'where is my order' support tickets.\n"
            f"- **Staffing:** if predicted times are trending up for a lane or "
            f"shipping mode, that's a signal to review warehouse processing "
            f"capacity or courier load before it becomes a backlog.\n"
            f"- **Exception handling:** orders predicted well above the norm "
            f"for their shipping mode (e.g. Express predicted near Standard "
            f"speeds) are good candidates for manual review or courier "
            f"reassignment."
        )

st.divider()
st.caption(
    "Model: scikit-learn Pipeline (StandardScaler + OneHotEncoder → "
    "LinearRegression), trained on log(Delivery_Time_Hours). "
    "For a university ML assignment — not a production forecasting tool."
)
