import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score

st.page_link("dashboard.py", label="Dashboard", icon="🏠")
st.page_link("streamlit_app.py", label="Diamond Prediction", icon="💎")
st.markdown("""
    <style>
    .nav-bar {
        background-color: #222;
        padding: 10px 0;
        text-align: center;
        margin-bottom: 20px;
    }
    .nav-bar a {
        color: #fff;
        margin: 0 20px;
        font-size: 1.1em;
        text-decoration: none;
        font-weight: bold;
    }
    .nav-bar a:hover {
        color: #1f77b4;
        text-decoration: underline;
    }
    </style>
    <div class="nav-bar">
        <a href="/dashboard" target="_self">Dashboard</a>
        <a href="/streamlit_app" target="_self">Diamond Prediction</a>
    </div>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Diamond Price Dashboard", layout="wide")

# --- Load Data & Model ---
df = pd.read_csv("diamonds.csv")
if 'Unnamed: 0' in df.columns:
    df = df.drop(['Unnamed: 0'], axis=1)
df = df[(df['x'] != 0) & (df['y'] != 0) & (df['z'] != 0)]
df['volume'] = df['x'] * df['y'] * df['z']

le_cut = joblib.load("le_cut.joblib")
le_color = joblib.load("le_color.joblib")
le_clarity = joblib.load("le_clarity.joblib")
scaler = joblib.load("scaler.joblib")
model = joblib.load("rf_model.joblib")

# --- Prepare Data for Model ---
df_enc = df.copy()
df_enc['cut'] = le_cut.transform(df_enc['cut'])
df_enc['color'] = le_color.transform(df_enc['color'])
df_enc['clarity'] = le_clarity.transform(df_enc['clarity'])
X = df_enc.drop(['price'], axis=1)
y = df_enc['price']
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
residuals = y - y_pred

# --- KPI Cards ---
avg_price = df['price'].mean()
med_carat = df['carat'].median()
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.markdown("""
    <style>
    .main-title {font-size:2.2em; font-weight:bold; text-align:center;}
    .subtitle {font-size:1.2em; text-align:center;}
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-title">DIAMOND PRICE PREDICT</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Dashboard</div>', unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Average Price", f"${avg_price:,.2f}")
kpi2.metric("Median Carat", f"{med_carat:.2f}")
kpi3.metric("Model MAE", f"${mae:,.2f}")
kpi4.metric("Model R²", f"{r2:.3f}")

st.markdown("---")

# --- Actual vs Predicted Scatter ---
st.subheader("Actual vs Predicted Price")
fig1 = px.scatter(x=y, y=y_pred, labels={'x': 'Actual Price', 'y': 'Predicted Price'}, opacity=0.5)
fig1.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode='lines', line=dict(color='red', dash='dash'), name='45° Line'))
st.plotly_chart(fig1, use_container_width=True)

# --- Feature Importance ---
st.subheader("Feature Importance")
importances = model.feature_importances_
features = X.columns
imp_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False)
fig2 = px.bar(imp_df, x='importance', y='feature', orientation='h', title="Random Forest Feature Importance")
st.plotly_chart(fig2, use_container_width=True)

# --- Price vs Carat Scatter/Hexbin ---
st.subheader("Price vs Carat")
hue = st.selectbox("Color by", ["None", "color", "clarity"])
if hue == "None":
    fig3 = px.scatter(df, x="carat", y="price", opacity=0.5)
else:
    fig3 = px.scatter(df, x="carat", y="price", color=hue, opacity=0.5)
st.plotly_chart(fig3, use_container_width=True)

# --- Category Effects ---
st.subheader("Category Effects on Price")
cat_col = st.selectbox("Show mean price by", ["cut", "color", "clarity"])
cat_df = df.groupby(cat_col)['price'].mean().reset_index()
fig4 = px.bar(cat_df, x=cat_col, y="price", title=f"Mean Price by {cat_col.capitalize()}")
st.plotly_chart(fig4, use_container_width=True)

# --- Residual Diagnostics ---
st.subheader("Residual Diagnostics")
tab1, tab2 = st.tabs(["Residuals vs Fitted", "Error by Carat Bin"])

with tab1:
    fig5 = px.scatter(x=y_pred, y=residuals, labels={'x': 'Fitted Price', 'y': 'Residuals'}, opacity=0.5)
    fig5.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig5, use_container_width=True)

with tab2:
    bins = pd.cut(df['carat'], bins=10)
    err_by_bin = pd.DataFrame({'carat_bin': bins, 'abs_error': np.abs(residuals)})
    err_bin_df = err_by_bin.groupby('carat_bin')['abs_error'].mean().reset_index()
    err_bin_df['carat_bin'] = err_bin_df['carat_bin'].astype(str)  # Convert Interval to string
    fig6 = px.bar(err_bin_df, x='carat_bin', y='abs_error', title="Mean Absolute Error by Carat Bin")
    st.plotly_chart(fig6, use_container_width=True)
    