import streamlit as st
import pandas as pd
import joblib

st.title("Diamond Price Prediction App")

# Load encoders, scaler, and model
le_cut = joblib.load("le_cut.joblib")
le_color = joblib.load("le_color.joblib")
le_clarity = joblib.load("le_clarity.joblib")
scaler = joblib.load("scaler.joblib")
model = joblib.load("rf_model.joblib")

# Load data for min/max values
df = pd.read_csv("diamonds.csv")
if 'Unnamed: 0' in df.columns:
    df = df.drop(['Unnamed: 0'], axis=1)
df = df[(df['x'] != 0) & (df['y'] != 0) & (df['z'] != 0)]
df['volume'] = df['x'] * df['y'] * df['z']

# Cap outliers for UI consistency
for col in ['volume', 'depth', 'table']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr
    df[col] = df[col].clip(lower=lower_whisker, upper=upper_whisker)

st.header("Enter Diamond Features")
carat = st.number_input("Carat", min_value=float(df['carat'].min()), max_value=float(df['carat'].max()), value=float(df['carat'].mean()))
cut = st.selectbox("Cut", options=le_cut.classes_)
color = st.selectbox("Color", options=le_color.classes_)
clarity = st.selectbox("Clarity", options=le_clarity.classes_)
depth = st.number_input("Depth", min_value=float(df['depth'].min()), max_value=float(df['depth'].max()), value=float(df['depth'].mean()))
table = st.number_input("Table", min_value=float(df['table'].min()), max_value=float(df['table'].max()), value=float(df['table'].mean()))
x = st.number_input("x (mm)", min_value=float(df['x'].min()), max_value=float(df['x'].max()), value=float(df['x'].mean()))
y_ = st.number_input("y (mm)", min_value=float(df['y'].min()), max_value=float(df['y'].max()), value=float(df['y'].mean()))
z = st.number_input("z (mm)", min_value=float(df['z'].min()), max_value=float(df['z'].max()), value=float(df['z'].mean()))

# Prepare input for prediction
input_dict = {
    'carat': carat,
    'cut': le_cut.transform([cut])[0],
    'color': le_color.transform([color])[0],
    'clarity': le_clarity.transform([clarity])[0],
    'depth': depth,
    'table': table,
    'x': x,
    'y': y_,
    'z': z,
    'volume': x * y_ * z
}
input_df = pd.DataFrame([input_dict])

# Apply MinMaxScaler to input
input_df_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Price"):
    price_pred = model.predict(input_df_scaled)[0]
    st.success(f"Predicted Diamond Price: ${price_pred:,.2f}")