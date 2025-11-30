import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from data_processing import classify_wefe_stress

# Page configuration
st.set_page_config(page_title="WEFE Stress Visualization", layout="wide")

# Load data
@st.cache_data
def load_data():
    """Load the WEFE stressed data."""
    df = pd.read_csv('Litani_WEFE_Stressed_Data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

# Load models
@st.cache_resource
def load_models():
    """Load the trained models."""
    lr_model = pickle.load(open('linear_regression_model.pkl', 'rb'))
    rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))
    return lr_model, rf_model

# Load data and models
try:
    df = load_data()
    lr_model, rf_model = load_models()
except Exception as e:
    st.error(f"Error loading data or models: {e}")
    st.stop()

# Title
st.title("WEFE Stress Index Visualization and Prediction")

# ========== FOUR GRAPHS EXACTLY AS IN VISUALIZATION.PY ==========
st.header("Data Visualizations")

# Prepare data for all charts
ndvi = df[["date", "ndvi"]].dropna()
ndvi["ndvi_roll30"] = ndvi["ndvi"].rolling(30, min_periods=1).mean()
ndvi["month"] = ndvi["date"].dt.month
seasonal_ndvi = ndvi.groupby("month")["ndvi"].mean()

ndti = df[["date", "ndti"]].dropna()
ndti["ndti_roll30"] = ndti["ndti"].rolling(30, min_periods=1).mean()

viirs = df[["date", "viirs_proxy"]].dropna()
viirs["viirs_roll30"] = viirs["viirs_proxy"].rolling(30, min_periods=1).mean()

df_rain = df.copy()
df_rain['rain_roll30'] = df_rain['rainfall'].rolling(30, min_periods=1).mean()

# Top row: NDVI and NDTI
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    # Graph 1: NDVI (2 subplots)
    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 4))
    ax1[0].plot(ndvi["date"], ndvi["ndvi_roll30"], color="forestgreen", lw=1.5)
    ax1[0].set(title="NDVI 30-Day Rolling Mean", xlabel="Date", ylabel="NDVI")

    ax1[1].bar(seasonal_ndvi.index, seasonal_ndvi.values, color="mediumseagreen")
    ax1[1].set_xticks(range(1, 13))
    ax1[1].set(title="Average Seasonal NDVI Cycle", xlabel="Month", ylabel="NDVI")
    plt.tight_layout()
    st.pyplot(fig1)

with row1_col2:
    # Graph 2: NDTI
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(ndti["date"], ndti["ndti_roll30"], color="chocolate", lw=1.5)
    ax2.set_title("NDTI (Turbidity Index) – 30-Day Rolling Mean")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("NDTI")
    plt.tight_layout()
    st.pyplot(fig2)

# Bottom row: VIIRS and Rainfall
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    # Graph 3: VIIRS
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(viirs["date"], viirs["viirs_roll30"], color="green", lw=1.5)
    ax3.set_title("VIIRS Reflectance (B02 Proxy) – 30-Day Rolling Mean")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Reflectance")
    ax3.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig3)

with row2_col2:
    # Graph 4: Rainfall
    plt.style.use("seaborn-v0_8-whitegrid")
    fig4, ax4 = plt.subplots(figsize=(10, 4))

    ax4.fill_between(
        df_rain["date"], df_rain["rain_roll30"],
        color="skyblue", alpha=0.35, label="Rainfall (mm)"
    )

    ax4.plot(
        df_rain["date"], df_rain["rain_roll30"],
        color="skyblue", linewidth=1.5, alpha=0.6
    )

    ax4.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Rainfall (mm)", color="blue", fontsize=12, fontweight="bold")
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4.grid(True, linestyle="--", alpha=0.4)

    fig4.suptitle(
        "Rainfall (30-Day Rolling Mean)",
        fontsize=16, fontweight="bold", y=1.03
    )

    ax4.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig4)

# ========== INTERACTIVE CONTROL PANEL ==========
st.header("Interactive Prediction Panel")

# Get feature ranges for sliders
ndvi_min, ndvi_max = float(df['ndvi'].min()), float(df['ndvi'].max())
ndti_min, ndti_max = float(df['ndti'].min()), float(df['ndti'].max())
rainfall_min, rainfall_max = float(df['rainfall'].min()), float(df['rainfall'].max())
viirs_min, viirs_max = float(df['viirs_proxy'].min()), float(df['viirs_proxy'].max())

# Create sliders
col1, col2 = st.columns(2)

with col1:
    st.subheader("Feature Controls")
    ndvi_val = st.slider(
        "NDVI",
        min_value=ndvi_min,
        max_value=ndvi_max,
        value=float(df['ndvi'].mean()),
        step=0.001,
        format="%.3f"
    )
    
    ndti_val = st.slider(
        "NDTI",
        min_value=ndti_min,
        max_value=ndti_max,
        value=float(df['ndti'].mean()),
        step=0.001,
        format="%.3f"
    )

with col2:
    st.subheader("")  # Empty header for alignment
    rainfall_val = st.slider(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=15.0,
        value=min(15.0, float(df['rainfall'].mean())),
        step=0.1,
        format="%.2f"
    )
    
    viirs_val = st.slider(
        "VIIRS Proxy",
        min_value=2000.0,
        max_value=6000.0,
        value=max(2000.0, min(6000.0, float(df['viirs_proxy'].mean()))),
        step=1.0,
        format="%.2f"
    )

# Make predictions using both trained models
# Create DataFrame with proper feature names to avoid warnings
input_features_df = pd.DataFrame({
    'ndvi': [ndvi_val],
    'ndti': [ndti_val],
    'rainfall': [rainfall_val],
    'viirs_proxy': [viirs_val]
})

# Predict using Linear Regression model
lr_prediction = lr_model.predict(input_features_df)[0]
# Predict using Random Forest model
rf_prediction = rf_model.predict(input_features_df)[0]

# Classify predictions
lr_category = classify_wefe_stress(lr_prediction)
rf_category = classify_wefe_stress(rf_prediction)

# Display predictions
st.subheader("Model Predictions")

pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    st.markdown("### Linear Regression Model")
    st.metric("Predicted WEFE Stress Index", f"{lr_prediction:.4f}")
    
    # Category display with color
    if lr_category == 'Red':
        st.error(f"Category: {lr_category}")
    elif lr_category == 'Amber':
        st.warning(f"Category: {lr_category}")
    else:
        st.success(f"Category: {lr_category}")

with pred_col2:
    st.markdown("### Random Forest Model")
    st.metric("Predicted WEFE Stress Index", f"{rf_prediction:.4f}")
    
    # Category display with color
    if rf_category == 'Red':
        st.error(f"Category: {rf_category}")
    elif rf_category == 'Amber':
        st.warning(f"Category: {rf_category}")
    else:
        st.success(f"Category: {rf_category}")

# Display current feature values
st.subheader("Current Feature Values")
feature_df = pd.DataFrame({
    'Feature': ['NDVI', 'NDTI', 'Rainfall (mm)', 'VIIRS Proxy'],
    'Value': [ndvi_val, ndti_val, rainfall_val, viirs_val]
})
st.dataframe(feature_df, width='stretch', hide_index=True)

