import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diamond Price Predictor", layout="centered")
st.title("💎 Diamond Price Prediction App")

@st.cache_resource
def load_model():
    return joblib.load("diamond_price_pipeline.joblib")

bundle = load_model()
pipe = bundle["pipe"]
features = bundle["features"]

# Define categorical orders (for nicer UI selection)
cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order = ["J", "I", "H", "G", "F", "E", "D"]
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

st.header("Enter Diamond Features")

col1, col2, col3 = st.columns(3)

with col1:
    carat = st.number_input("Carat", min_value=0.1, max_value=5.0, value=1.0, step=0.01)
    cut = st.selectbox("Cut", options=cut_order)
    color = st.selectbox("Color", options=color_order)

with col2:
    clarity = st.selectbox("Clarity", options=clarity_order)
    depth = st.slider("Depth (%)", 50.0, 75.0, 61.5)
    table = st.slider("Table (%)", 50.0, 70.0, 57.0)

with col3:
    x = st.number_input("x (mm)", min_value=2.0, max_value=12.0, value=5.0, step=0.01)
    y = st.number_input("y (mm)", min_value=2.0, max_value=12.0, value=5.0, step=0.01)
    z = st.number_input("z (mm)", min_value=2.0, max_value=12.0, value=3.0, step=0.01)

# Build input dataframe
input_dict = {
    "carat": carat,
    "cut": cut,
    "color": color,
    "clarity": clarity,
    "depth": depth,
    "table": table,
    "x": x,
    "y": y,
    "z": z,
    "volume": x * y * z
}
input_df = pd.DataFrame([input_dict])

# Predict
if st.button("💰 Predict Price"):
    price_pred = pipe.predict(input_df)[0]
    st.success(f"Predicted Diamond Price: ${price_pred:,.2f}")
