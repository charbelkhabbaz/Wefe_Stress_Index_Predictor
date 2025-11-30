import pandas as pd
import numpy as np
import os

# Define the names of the uploaded CSV files
FILES = {
    'ndvi': 'Litani_NDVI_Daily_Data.csv',
    'ndti': 'Litani_NDTI_Daily_Data.csv',
    'rainfall': 'Litani_Rainfall_Daily_Sum_Data.csv',
    'viirs_proxy': 'Litani_VIIRS_Proxy_MODIS_NIR_Daily_Data.csv'
}

# Define the data columns to use for merging and calculation
COLUMN_MAP = {
    'ndvi': 'ndvi_mean',
    'ndti': 'ndti_mean',
    'rainfall': 'precipitation_sum',
    'viirs_proxy': 'sur_refl_b02_mean'
}

# 1. Classification function
def classify_wefe_stress(index):
    """Classifies the WEFE stress index into Green, Amber, or Red."""
    if index >= 0.7:
        return 'Red'
    elif index >= 0.4:
        return 'Amber'
    else:
        return 'Green'

# 2. Function to load and merge data
def load_and_merge_data(files_map):
    """Loads CSV files, renames columns, and merges them on the 'date' column."""
    merged_df = None
    
    for key, filename in files_map.items():
        if not os.path.exists(filename):
            print(f"Error: File not found: {filename}. Please ensure it is in the current directory.")
            return None

        # Load the CSV, ensuring 'date' is parsed correctly and handling potential NA values
        df = pd.read_csv(filename)
        
        # Rename the value column and select only 'date' and the value column
        value_col = COLUMN_MAP[key]
        df = df[['date', value_col]].rename(columns={value_col: key})
        
        # Initialize or merge the DataFrame
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='date', how='outer')
            
    # Convert 'date' to datetime objects and sort
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    
    # Fill any missing data with the mean of that column (Imputation)
    print("Handling missing values (NaN) by filling with the column mean.")
    for col in COLUMN_MAP.keys():
        merged_df[col].fillna(merged_df[col].mean(), inplace=True)
        
    return merged_df

# Helper function for anomaly-robust normalization (extracted from main function)
def _apply_robust_normalization(series, col_name):
    """
    Applies Min-Max normalization using robust bounds (1st percentile min) and 
    choosing the safest max cap between the 99th percentile and the 10x outlier rule.
    Returns the normalized Series (0 to 1).
    """
    
    # 1. Calculate the 1st percentile for the minimum normalization bound (robust X_min)
    min_quantile = series.quantile(0.01)

    # 2. Maximum Anomaly Handling Logic 
    
    # Cap A: The standard robust cap (99th percentile)
    cap_99th_percentile = series.quantile(0.99)
    
    # Cap B: The conditional cap based on the 10x rule
    cap_10x_rule = cap_99th_percentile # Start with 99th percentile as default for 10x rule cap

    unique_sorted_values = np.sort(series.dropna().unique())[::-1]
    
    if len(unique_sorted_values) >= 2:
        X1 = unique_sorted_values[0] # Largest value
        X2 = unique_sorted_values[1] # Second largest value
        
        # If the largest value is significantly larger than the second largest...
        if X1 > 10 * X2:
            # Set the conditional cap to the second largest value (X2)
            print(f"  --> ALERT: {col_name} Max value ({X1:.2f}) is > 10x the next ({X2:.2f}). Capping max at X2.")
            cap_10x_rule = X2
        
    # 3. Final Determination of Max Cap: Choose the smallest cap for best resolution
    max_quantile = min(cap_10x_rule, cap_99th_percentile)
    
    # 4. Apply Normalization
    
    # Check if the range is zero (safety check)
    if (max_quantile - min_quantile) == 0:
        return series.apply(lambda x: 0.5) # Set to mid-point if no variation

    # Clamp the raw data to the determined bounds (min_quantile and max_quantile)
    clamped_data = series.clip(lower=min_quantile, upper=max_quantile)
    
    # Apply the Min-Max Formula using the determined thresholds
    normalized_series = (clamped_data - min_quantile) / (max_quantile - min_quantile)
    
    return normalized_series


# 3. Normalization and Stress Index Calculation
def calculate_wefe_stress_index(df):
    """
    Normalizes features, applies a randomized weight tweak, and 
    calculates the WEFE Stress Index.
    """
    
    # a. Robust Normalization (Min-Max Scaling) using the new helper function
    df_normalized = df[['ndvi', 'ndti', 'rainfall', 'viirs_proxy']].copy()
    
    print("Applying robust Min-Max normalization with conditional 10x outlier handling.")

    for col in df_normalized.columns:
        df_normalized[col] = _apply_robust_normalization(df[col], col)

    # b. Apply Stress Direction (Inversion)
    # The stress index assumes high values mean high stress (Red).
    
    # Features where high value means LOW stress, so we invert them:
    for col in ['ndvi', 'rainfall', 'viirs_proxy']:
        df_normalized[col + '_stress'] = 1 - df_normalized[col]
    
    # Feature where high value means HIGH stress (NDTI):
    df_normalized['ndti_stress'] = df_normalized['ndti']

    # c. Calculate Stress Index with TWEAKS
    
    # Base weight for each feature (25%)
    BASE_WEIGHT = 0.25
    
    # Define the core features for the stress index
    stress_features = ['ndvi_stress', 'ndti_stress', 'rainfall_stress', 'viirs_proxy_stress']
    
    # Create randomized weights (tweaks) for each observation to prevent cheating.
    # The tweak is +/- 0.1% (0.001) of the base 25% weight (0.25).
    # Tweak range: BASE_WEIGHT * (1 +/- 0.001) -> [0.24975, 0.25025]
    
    # Generate random adjustments for each feature for every row
    tweak = np.random.uniform(low=-0.00025, high=0.00025, size=(len(df_normalized), 4)) # 4 columns
    
    # Calculate the tweaked weights for all rows
    tweaked_weights = BASE_WEIGHT + tweak
    
    # Ensure the sum of tweaked weights remains close to 1 (or 4 * BASE_WEIGHT)
    # Re-normalize weights so their sum is exactly 1 for each row
    tweaked_weights_sum = np.sum(tweaked_weights, axis=1, keepdims=True)
    final_weights = tweaked_weights / tweaked_weights_sum
    
    print(f"Applying randomized weights to each feature to prevent cheating (e.g., NDVI weight varies around 25%).")

    # Multiply normalized stress values by their respective final weights
    weighted_stress = (
        df_normalized['ndvi_stress'].values * final_weights[:, 0] +
        df_normalized['ndti_stress'].values * final_weights[:, 1] +
        df_normalized['rainfall_stress'].values * final_weights[:, 2] +
        df_normalized['viirs_proxy_stress'].values * final_weights[:, 3]
    )

    # The result is the final WEFE Stress Index (0 to 1)
    df['wefe_stress_index'] = weighted_stress
    
    return df

# Main execution block
if __name__ == '__main__':
    print("--- WEFE Stress Index Generator ---")
    
    # 1. Load and Merge Data
    merged_df = load_and_merge_data(FILES)
    
    if merged_df is None:
        exit()
    
    # 2. Calculate Stress Index
    final_df = calculate_wefe_stress_index(merged_df)
    
    # 3. Classify Stress Index
    final_df['category'] = final_df['wefe_stress_index'].apply(classify_wefe_stress)
    
    # 4. Display and Save Results
    output_filename = 'Litani_WEFE_Stressed_Data.csv'
    
    # Select final columns for output (date, raw data, index, category)
    output_cols = ['date'] + list(COLUMN_MAP.keys()) + ['wefe_stress_index', 'category']
    final_df[output_cols].to_csv(output_filename, index=False)
    
    print("\n--- RESULTS ---")
    print(f"Combined data points: {len(final_df)}")
    print(f"Final data saved to: {output_filename}")
    print("\nSample of the final data table:")
    print(final_df[['date', 'wefe_stress_index', 'category']].head())
    print(final_df[['date', 'wefe_stress_index', 'category']].tail())