import streamlit as st
import pandas as pd
import joblib

st.title("Diamond Price Prediction App")

# Load the pipeline bundle
@st.cache_data
def load_model():
    try:
        bundle = joblib.load("diamond_price_pipeline.joblib")
        return bundle
    except FileNotFoundError:
        st.error("Model file not found. Please run the training script first: `python random_forests.py`")
        st.stop()

bundle = load_model()
pipeline = bundle["pipe"]

# Load data for min/max values (same logic as training)
@st.cache_data
def load_data():
    df = pd.read_csv("diamonds.csv")
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]
    df['volume'] = df['x'] * df['y'] * df['z']
    
    # Cap outliers for UI consistency (same as training)
    for col in ['volume', 'depth', 'table']:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        df[col] = df[col].clip(low, high)
    
    return df

df = load_data()

st.header("Enter Diamond Features")

# Input widgets with proper order (matching training pipeline)
carat = st.number_input("Carat", min_value=float(df['carat'].min()), max_value=float(df['carat'].max()), value=float(df['carat'].mean()))

cut = st.selectbox("Cut", options=["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", options=["J", "I", "H", "G", "F", "E", "D"])
clarity = st.selectbox("Clarity", options=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

depth = st.slider("Depth", min_value=float(df['depth'].min()), max_value=float(df['depth'].max()), value=float(df['depth'].mean()))
table = st.slider("Table", min_value=float(df['table'].min()), max_value=float(df['table'].max()), value=float(df['table'].mean()))
x = st.slider("x (mm)", min_value=float(df['x'].min()), max_value=float(df['x'].max()), value=float(df['x'].mean()))
y_ = st.slider("y (mm)", min_value=float(df['y'].min()), max_value=float(df['y'].max()), value=float(df['y'].mean()))
z = st.slider("z (mm)", min_value=float(df['z'].min()), max_value=float(df['z'].max()), value=float(df['z'].mean()))

# Predict using the pipeline (handles all preprocessing automatically)
if st.button("Predict Price"):
    try:
        # Create input dataframe (same format as training)
        input_data = pd.DataFrame({
            'cut': [cut],
            'color': [color], 
            'clarity': [clarity],
            'carat': [carat],
            'depth': [depth],
            'table': [table],
            'x': [x],
            'y': [y_],
            'z': [z],
            'volume': [x * y_ * z]
        })
        
        price_pred = pipeline.predict(input_data)[0]
        st.success(f"Predicted Diamond Price: ${price_pred:,.2f}")
        
    except Exception as e:
        st.error(f"Prediction error: {e}")