import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Diamond Price Dashboard", layout="wide")

# ============ Load bundle (pipeline + metadata) ============
# This is the artifact saved by the training script:
# joblib.dump({"pipe": pipe, "log_target": LOG_TARGET, "features": CATEG+NUM}, "diamond_price_pipeline.joblib")
bundle = joblib.load("diamond_price_pipeline.joblib")
pipe = bundle["pipe"]
LOG_TARGET = bundle.get("log_target", False)
features = bundle["features"]  # feature order the model expects

# ============ Load & minimally clean data (match training) ============
df = pd.read_csv("diamonds.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# drop impossible dims, create volume
df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]
df["volume"] = df["x"] * df["y"] * df["z"]

# IQR capping for the same numeric cols used during training
for col in ["volume", "depth", "table"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    df[col] = df[col].clip(low, high)

# ============ Build X/y, predict, compute residuals ============
X = df[features].copy()
y = df["price"].copy()

def inv(t):
    return np.expm1(t) if LOG_TARGET else t

y_pred = inv(pipe.predict(X))
residuals = y - y_pred

# ============ Header & KPIs ============
st.markdown(
    """
    <style>
      .main-title {font-size:2.2em; font-weight:bold; text-align:center;}
      .subtitle {font-size:1.2em; text-align:center; margin-bottom: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="main-title">DIAMOND PRICE PREDICT</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Dashboard</div>', unsafe_allow_html=True)

avg_price = df["price"].mean()
med_carat = df["carat"].median()
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Average Price", f"${avg_price:,.2f}")
k2.metric("Median Carat", f"{med_carat:.2f}")
k3.metric("Model MAE", f"${mae:,.2f}")
k4.metric("Model R²", f"{r2:.3f}")

st.markdown("---")

# ============ Actual vs Predicted ============
st.subheader("Actual vs Predicted Price")

xmin, xmax = float(y.min()), float(y.max())
fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=y, y=y_pred, mode="markers",
        marker=dict(color="lightskyblue", opacity=0.55, size=5),
        name="Predicted"
    )
)
fig1.add_trace(
    go.Scatter(
        x=[xmin, xmax], y=[xmin, xmax],
        mode="lines", line=dict(color="red", width=2, dash="dash"),
        name="45° Line"
    )
)
fig1.update_layout(xaxis_title="Actual Price", yaxis_title="Predicted Price")
st.plotly_chart(fig1, use_container_width=True)

# ============ Feature Importance ============
st.subheader("Feature Importance")

# Try to get the transformed feature names (works with ColumnTransformer)
try:
    feat_names = pipe.named_steps["prep"].get_feature_names_out()
except Exception:
    feat_names = np.array(features)

try:
    importances = pipe.named_steps["model"].feature_importances_
    imp_df = (
        pd.DataFrame({"feature": feat_names, "importance": importances})
        .sort_values("importance", ascending=False)
    )
    fig2 = px.bar(imp_df.head(20), x="importance", y="feature", orientation="h",
                  title="Top Feature Importances")
    st.plotly_chart(fig2, use_container_width=True)
except Exception:
    st.info("Current model does not expose feature_importances_.")

# ============ Category Effects ============
st.subheader("Category Effects on Price")
cat_col = st.selectbox("Show mean price by", ["cut", "color", "clarity"])
cat_df = df.groupby(cat_col, as_index=False)["price"].mean()
fig4 = px.bar(cat_df, x=cat_col, y="price", title=f"Mean Price by {cat_col.capitalize()}")
st.plotly_chart(fig4, use_container_width=True)

# ============ Price Elasticity by Carat Band and Cut ============
st.subheader("Price Elasticity by Carat Band and Cut")

# Create carat bands
bins = [0.2, 0.5, 1.0, 2.0, df['carat'].max()+1]
labels = ['0.2–0.5 ct','0.5–1.0 ct','1.0–2.0 ct','2.0+ ct']
df['carat_band'] = pd.cut(df['carat'], bins=bins, labels=labels, include_lowest=True)

from plotly.subplots import make_subplots

cuts = df['cut'].unique()
bands = df['carat_band'].cat.categories

fig = make_subplots(rows=2, cols=2, subplot_titles=bands)

for i, band in enumerate(bands):
    row, col = divmod(i, 2)
    band_df = df[df['carat_band'] == band]
    for cut in cuts:
        cut_df = band_df[band_df['cut'] == cut]
        fig.add_trace(
            go.Scatter(
                x=cut_df['carat'],
                y=cut_df['price'],
                mode='markers',
                marker=dict(size=5, opacity=0.25),
                name=str(cut),
                legendgroup=str(cut),
                showlegend=(i==0)
            ),
            row=row+1, col=col+1
        )
        # Regression line (simple linear fit)
        if len(cut_df) > 1:
            fit = np.polyfit(cut_df['carat'], cut_df['price'], 1)
            x_fit = np.linspace(cut_df['carat'].min(), cut_df['carat'].max(), 50)
            y_fit = fit[0]*x_fit + fit[1]
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode='lines',
                    line=dict(width=2),
                    name=f"{cut} fit",
                    legendgroup=str(cut),
                    showlegend=False
                ),
                row=row+1, col=col+1
            )

fig.update_layout(
    height=800, width=1000,
    title_text="Price Elasticity by Carat Band and Cut",
    legend_title_text="Cut"
)
st.plotly_chart(fig, use_container_width=True)