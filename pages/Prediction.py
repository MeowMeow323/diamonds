import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diamond Price Predictor", layout="centered")
st.title("💎 Diamond Price Prediction App")

# Diamond Features Information Section
with st.expander("ℹ️ Learn About Diamond Features"):
    st.markdown("""
    ### Diamond Quality Factors (The 4 Cs + Dimensions)
    
    **🔸 Carat Weight**
    - Measures the diamond's weight (1 carat = 200 milligrams)
    - Larger diamonds are rarer and more valuable
    - Range: 0.1 - 5.0 carats (typical)
    """)
    
    st.markdown("""
    **✂️ Cut Quality**
    - Refers to how well the diamond has been cut and polished
    - Affects brilliance and sparkle
    - Order (worst to best): Fair → Good → Very Good → Premium → Ideal
    """)
    
    st.markdown("""
    **🎨 Color Grade**
    - Measures the absence of color in white diamonds
    - Order (most color to colorless): J → I → H → G → F → E → D
    - D is completely colorless and most valuable
    """)
    
    # Diamond color grade image
    st.image("assets/diamond_color.jpg", 
             caption="Diamond color grades from J (tinted) to D (colorless)", width=675)
    
    st.markdown("""
    **🔍 Clarity Grade**
    - Measures internal flaws (inclusions) and surface blemishes
    - Order (most flaws to flawless): I1 → SI2 → SI1 → VS2 → VS1 → VVS2 → VVS1 → IF
    - IF (Internally Flawless) is the highest grade shown
    """)
    
    # Diamond clarity image
    st.image("assets/diamond_clarity.jpg", 
             caption="Diamond clarity - fewer inclusions mean higher grades", width=675)
    
    st.markdown("""
    **📏 Physical Dimensions**
    - **Depth %**: Total depth percentage (depth/average diameter × 100)
    - **Table %**: Width of the top facet relative to the widest point
    - **x, y, z (mm)**: Length, width, and depth measurements
    - **Volume**: Calculated as x × y × z, affects the diamond's visual size
    """)
    
    # Diamond proportions diagram
    st.image("https://images.unsplash.com/photo-1596944946317-7c07d4ea2b5d?w=600&h=300&fit=crop", 
             caption="Diamond proportions: depth, table, and dimensions affect appearance", width=400)
    
    st.markdown("*Tip: Well-balanced proportions enhance a diamond's beauty and value!*")

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
