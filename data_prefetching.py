# This script fetches four environmental datasets from Google Earth Engine:
# Daily NDVI, Daily NDTI, Daily Rainfall (CHIRPS), and a proxy for DAILY VIIRS using MODIS NIR (Band 2).

# ==============================================================================
# CONFIRMING SCRIPT VERSION
# ==============================================================================

import ee
import datetime
import warnings

# Suppress the DeprecationWarning as it's not a hard error.
warnings.filterwarnings('ignore', category=DeprecationWarning, module='ee.deprecation')

# The unique project ID from the Canvas environment.
project = 'litani-basin-wefe'

# ==============================================================================
# 1. AUTHENTICATE AND INITIALIZE EARTH ENGINE
# ==============================================================================
try:
    # Use the specified project ID
    ee.Initialize(project=project) 
    print("Google Earth Engine has been initialized successfully.")
except Exception as e:
    print(f"Error initializing GEE: {e}.")
    print("Please ensure you have authenticated your GEE account and have access to the project 'litani-basin-wefe'.")
    raise RuntimeError("GEE authentication failed. Please check your account permissions.")

# ==============================================================================
# 2. DEFINE THE REGION OF INTEREST (LITANI BASIN)
# ==============================================================================
litani_basin = ee.Geometry.Polygon(
    # Coordinates defining a rough bounding box for the Litani Basin
    [[[35.5, 33.2], [36.2, 33.2], [36.2, 34.1], [35.5, 34.1], [35.5, 33.2]]],
    None, False
)

# ==============================================================================
# 3. DEFINE THE TIME PERIOD
# ==============================================================================
start_date = '2014-01-01'
# Fetch data up to today
end_date = datetime.date.today().strftime('%Y-%m-%d') 

# ==============================================================================
# 4. DATA FETCHING FUNCTION (Daily is all we need now)
# ==============================================================================

def get_daily_time_series(dataset_id, bands, reducer, start, end, region, scale=500):
    """
    Extracts a daily time-series from a GEE ImageCollection.
    Aggregates the images that fall within each 24-hour period (day).
    """
    if isinstance(dataset_id, str):
        collection = ee.ImageCollection(dataset_id)
    else:
        collection = dataset_id

    collection = collection \
        .filterDate(start, end) \
        .filterBounds(region) \
        .select(bands)

    start_date_ee = ee.Date(start)
    end_date_ee = ee.Date(end)
    n_days = end_date_ee.difference(start_date_ee, 'day').toInt()
    
    def daily_aggregation(n):
        date = start_date_ee.advance(n, 'day')
        start_of_day = ee.Date(date)
        end_of_day = start_of_day.advance(1, 'day')
        
        # Reduce all images that fall within that one day
        daily_image = collection.filterDate(start_of_day, end_of_day).reduce(reducer)
        
        mean_value = daily_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=1e13
        )
        
        return ee.Feature(None, mean_value).set('date', start_of_day.format('YYYY-MM-dd'))

    daily_series = ee.FeatureCollection(
        ee.List.sequence(0, n_days.subtract(1)).map(daily_aggregation)
    )
    return daily_series

# ------------------------------------------------------------------------------
# 4.1. Normalized Difference Vegetation Index (NDVI - DAILY)
# ------------------------------------------------------------------------------
print("Fetching **Daily** Normalized Difference Vegetation Index (NDVI) data...")
def add_ndvi(image):
    # MODIS bands for NDVI: Near-Infrared (sur_refl_b02) and Red (sur_refl_b01)
    ndvi_value = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01']).rename('ndvi')
    return image.addBands(ndvi_value)

# Use MODIS daily surface reflectance collection
ndvi_collection = ee.ImageCollection('MODIS/061/MOD09GA') \
    .filterDate(start_date, end_date) \
    .filterBounds(litani_basin) \
    .map(add_ndvi) \
    .select('ndvi')

ndvi_daily_series = get_daily_time_series(
    dataset_id=ndvi_collection,
    bands=['ndvi'],
    reducer=ee.Reducer.mean(),
    start=start_date,
    end=end_date,
    region=litani_basin,
    scale=500 
)

# ------------------------------------------------------------------------------
# 4.2. Water Quality Proxy (NDTI - DAILY)
# ------------------------------------------------------------------------------
print("Fetching **Daily** water quality (NDTI) data...")
def add_ndti(image):
    # Normalized Difference Turbidity Index using MODIS Red (b04) and Blue (b01)
    return image.addBands(image.normalizedDifference(['sur_refl_b04', 'sur_refl_b01']).rename('ndti'))

# NDTI is calculated from the daily MODIS surface reflectance product
ndti_collection = ee.ImageCollection('MODIS/061/MOD09GA') \
    .filterDate(start_date, end_date) \
    .filterBounds(litani_basin) \
    .map(add_ndti) \
    .select('ndti')

ndti_daily_series = get_daily_time_series(
    dataset_id=ndti_collection,
    bands=['ndti'],
    reducer=ee.Reducer.mean(),
    start=start_date,
    end=end_date,
    region=litani_basin,
    scale=500
)

# ------------------------------------------------------------------------------
# 4.3. Rainfall (CHIRPS - DAILY SUM)
# ------------------------------------------------------------------------------
print("Fetching **Daily** rainfall data...")
rainfall_daily_sum_series = get_daily_time_series(
    dataset_id='UCSB-CHG/CHIRPS/DAILY',
    bands=['precipitation'],
    reducer=ee.Reducer.sum(), # Sum for total daily rainfall
    start=start_date,
    end=end_date,
    region=litani_basin,
    scale=5566 
)

# ------------------------------------------------------------------------------
# 4.4. VIIRS PROXY (DAILY MODIS NIR Band 2) - FIX FOR HTTP ERROR 400
# ------------------------------------------------------------------------------
# FIX: Replacing the restricted VIIRS asset with the publicly accessible MODIS NIR Band 2, 
# which is similar to the VIIRS M7 band and is available daily.
print("Fetching **Daily** VIIRS Proxy data (MODIS Near-Infrared Reflectance - Band 2).")

modis_nir_daily_series = get_daily_time_series(
    dataset_id='MODIS/061/MOD09GA',
    bands=['sur_refl_b02'], # MODIS Band 2 (NIR) as a daily proxy
    reducer=ee.Reducer.mean(),
    start=start_date,
    end=end_date,
    region=litani_basin,
    scale=500 
)

# ==============================================================================
# 5. EXPORT EACH DATASET TO A SEPARATE CSV
# ==============================================================================
print("\nExporting datasets to Google Drive...")

# Export Daily NDVI data
task_ndvi = ee.batch.Export.table.toDrive(
    collection=ndvi_daily_series,
    description='Litani_NDVI_Daily_Data',
    folder='GEE_Exports',
    fileNamePrefix='Litani_NDVI_Daily_Data',
    fileFormat='CSV'
)
task_ndvi.start()
print("Exporting **Daily** Normalized Difference Vegetation Index (NDVI) data. Check the GEE 'Tasks' tab.")

# Export Daily NDTI data
task_ndti = ee.batch.Export.table.toDrive(
    collection=ndti_daily_series,
    description='Litani_NDTI_Daily_Data',
    folder='GEE_Exports',
    fileNamePrefix='Litani_NDTI_Daily_Data',
    fileFormat='CSV'
)
task_ndti.start()
print("Exporting **Daily** water quality (NDTI) data. Check the GEE 'Tasks' tab.")

# Export Daily Rainfall data
task_rainfall = ee.batch.Export.table.toDrive(
    collection=rainfall_daily_sum_series,
    description='Litani_Rainfall_Daily_Sum_Data',
    folder='GEE_Exports',
    fileNamePrefix='Litani_Rainfall_Daily_Sum_Data',
    fileFormat='CSV'
)
task_rainfall.start()
print("Exporting **Daily** summed rainfall data. Check the GEE 'Tasks' tab.")

# Export Daily VIIRS Proxy data
task_viirs = ee.batch.Export.table.toDrive(
    collection=modis_nir_daily_series,
    description='Litani_VIIRS_Proxy_MODIS_NIR_Daily_Data', # Updated description
    folder='GEE_Exports',
    fileNamePrefix='Litani_VIIRS_Proxy_MODIS_NIR_Daily_Data', # Updated file name
    fileFormat='CSV'
)
task_viirs.start()
print("Exporting **Daily** VIIRS Proxy data (MODIS NIR) (FIXED). Check the GEE 'Tasks' tab.")


print("\nAll four daily export tasks have been started. This should now run successfully without the asset access error. You can monitor their progress in the Google Earth Engine Code Editor under the 'Tasks' tab.")