"""
Train and save both Linear Regression and Random Forest models.
Run this script first before using the Streamlit app.

This script will create:
- linear_regression_model.pkl
- random_forest_model.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Define the file name for the preprocessed data
INPUT_FILE = 'Litani_WEFE_Stressed_Data.csv'

def train_linear_regression(X_train, y_train, X_test, y_test):
    """Train Linear Regression model and return the trained model."""
    print("\n" + "="*60)
    print("Training Linear Regression Model...")
    print("="*60)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Print coefficients
    print("\nModel Coefficients:")
    coefficients = pd.Series(model.coef_, index=X_train.columns)
    print(coefficients.sort_values(ascending=False))
    
    return model

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model and return the trained model."""
    print("\n" + "="*60)
    print("Training Random Forest Model...")
    print("="*60)
    
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Print feature importances
    print("\nFeature Importances:")
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print(importances.sort_values(ascending=False))
    
    return model

def main():
    """Main function to train both models and save them."""
    print("="*60)
    print("WEFE Stress Index - Model Training Script")
    print("="*60)
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ Error: '{INPUT_FILE}' not found!")
        print("Please run 'python data_processing.py' first to generate the data file.")
        return
    
    # Load data
    print(f"\n📊 Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Define features and target
    feature_cols = ['ndvi', 'ndti', 'rainfall', 'viirs_proxy']
    target_col = 'wefe_stress_index'
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")
    print(f"Dataset size: {len(df)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {len(X_train)} samples (80%)")
    print(f"Testing set: {len(X_test)} samples (20%)")
    
    # Train Linear Regression
    lr_model = train_linear_regression(X_train, y_train, X_test, y_test)
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Save models
    print("\n" + "="*60)
    print("Saving Models...")
    print("="*60)
    
    lr_output = 'linear_regression_model.pkl'
    rf_output = 'random_forest_model.pkl'
    
    with open(lr_output, 'wb') as f:
        pickle.dump(lr_model, f)
    print(f"✅ Saved: {lr_output}")
    
    with open(rf_output, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"✅ Saved: {rf_output}")
    
    # Check file sizes
    lr_size = os.path.getsize(lr_output) / (1024 * 1024)
    rf_size = os.path.getsize(rf_output) / (1024 * 1024)
    
    print(f"\n📦 Model file sizes:")
    print(f"   Linear Regression: {lr_size:.2f} MB")
    print(f"   Random Forest: {rf_size:.2f} MB")
    
    print("\n" + "="*60)
    print("✅ Model training completed successfully!")
    print("="*60)
    print("\nYou can now run the Streamlit app:")
    print("   streamlit run streamlit_app.py")

if __name__ == '__main__':
    main()

