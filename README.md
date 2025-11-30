# WEFE Stress Index Visualization and Prediction

A Streamlit web application for visualizing and predicting Water-Energy-Food-Ecosystem (WEFE) stress indices using machine learning models.

## Overview

This project provides an interactive dashboard to:
- Visualize environmental data (NDVI, NDTI, Rainfall, VIIRS) with time series charts
- Predict WEFE stress indices using Linear Regression and Random Forest models
- Classify stress levels into Green, Amber, or Red categories

## Features

### 📊 Data Visualizations
- **NDVI (Normalized Difference Vegetation Index)**: 30-day rolling mean and seasonal cycle
- **NDTI (Normalized Difference Turbidity Index)**: 30-day rolling mean
- **VIIRS Reflectance**: 30-day rolling mean of B02 proxy data
- **Rainfall**: 30-day rolling mean with filled area visualization

### 🤖 Machine Learning Models
- **Linear Regression**: Fast and interpretable predictions
- **Random Forest**: High-accuracy ensemble predictions with feature importance

### 🎛️ Interactive Controls
- Real-time sliders for adjusting feature values
- Instant predictions from both models
- Color-coded category classification (Green/Amber/Red)

## Project Structure

```
wefe_stress_code/
├── data_processing.py          # Data loading, merging, and stress index calculation
├── ML.py                        # Linear Regression model training
├── train_random_forest.py      # Random Forest model training
├── streamlit_app.py            # Main Streamlit application
├── visualization.py             # Original visualization code
├── Litani_WEFE_Stressed_Data.csv    # Processed dataset
├── linear_regression_model.pkl      # Trained Linear Regression model
├── random_forest_model.pkl         # Trained Random Forest model
└── README.md                    # This file
```

## Prerequisites

- Python 3.7+
- Required packages (install via `pip install -r requirements.txt` or individually):
  - `streamlit`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `pickle` (built-in)

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install streamlit pandas numpy matplotlib scikit-learn
   ```

3. **Ensure data files are present:**
   - `Litani_NDVI_Daily_Data.csv`
   - `Litani_NDTI_Daily_Data.csv`
   - `Litani_Rainfall_Daily_Sum_Data.csv`
   - `Litani_VIIRS_Proxy_MODIS_NIR_Daily_Data.csv`

## Usage

### Step 1: Process the Data

Generate the processed dataset with WEFE stress indices:

```bash
python data_processing.py
```

This will create `Litani_WEFE_Stressed_Data.csv`.

### Step 2: Train the Models (if needed)

**Train Linear Regression model:**
```bash
python ML.py
```

**Train Random Forest model:**
```bash
python train_random_forest.py
```

This will create `linear_regression_model.pkl` and `random_forest_model.pkl`.

### Step 3: Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## How to Use the Application

1. **View Visualizations**: Scroll to see the four data visualization charts arranged in a 2x2 grid.

2. **Adjust Feature Values**: Use the sliders in the "Interactive Prediction Panel" to set:
   - **NDVI**: Range from data minimum to maximum
   - **NDTI**: Range from data minimum to maximum
   - **Rainfall**: 0 to 15 mm
   - **VIIRS Proxy**: 2000 to 6000

3. **View Predictions**: As you adjust the sliders, predictions from both models update automatically:
   - **Linear Regression Model**: Shows predicted WEFE stress index and category
   - **Random Forest Model**: Shows predicted WEFE stress index and category

4. **Category Classification**:
   - **Green**: Stress index < 0.4 (Low stress)
   - **Amber**: Stress index 0.4 - 0.7 (Moderate stress)
   - **Red**: Stress index ≥ 0.7 (High stress)

## Model Performance

### Linear Regression
- R² Score: ~0.95-0.97
- Provides interpretable coefficients for each feature

### Random Forest
- R² Score: ~0.97
- Feature Importance:
  - NDTI: ~57%
  - Rainfall: ~18%
  - NDVI: ~18%
  - VIIRS Proxy: ~6%

## Data Sources

The application uses daily environmental data from the Litani region:
- NDVI (Normalized Difference Vegetation Index)
- NDTI (Normalized Difference Turbidity Index)
- Rainfall (daily sum)
- VIIRS Proxy (MODIS NIR reflectance)

## WEFE Stress Index Calculation

The stress index is calculated using:
1. Robust normalization (Min-Max with outlier handling)
2. Stress direction inversion (high NDVI/Rainfall/VIIRS = low stress)
3. Weighted combination of normalized stress features
4. Classification into Green/Amber/Red categories

## Troubleshooting

**Error: Model file not found**
- Run `python train_random_forest.py` to regenerate the Random Forest model
- Run `python ML.py` to regenerate the Linear Regression model

**Error: Data file not found**
- Ensure all CSV data files are in the project directory
- Run `python data_processing.py` to generate the processed dataset

**App won't start**
- Check that Streamlit is installed: `pip install streamlit`
- Verify Python version is 3.7 or higher

## License

This project is provided as-is for research and educational purposes.



