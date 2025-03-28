import os
import pandas as pd
import geopandas as gpd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np  
from matplotlib.colors import LinearSegmentedColormap
import random
from statsmodels.tsa.arima.model import ARIMA  # For forecasting
from joblib import Parallel, delayed  # For parallel processing

expected_columns = [
    "STATION", "LATITUDE", "LONGITUDE", "ELEVATION", "DATE",
    "CDSD", "CLDD", "HTDD", "HDSD", "DT00", "DT32", "DX32", "DX70", "DX90",
    "DP01", "DP10", "DSND", "DSNW", "EMXP", "EMSD", "EMSN", "PRCP", "SNOW",
    "TAVG", "TMAX", "TMIN", "EMXT", "EMNT", "AWND", "WDF2", "WDF5", "WSF2", "WSF5",
    "PSUN"
]

# -------------------------------
# Data Processing Functions
# -------------------------------
def clean_and_standardize(df):
    df = df[[col for col in df.columns if col in expected_columns]]
    missing_columns = set(expected_columns) - set(df.columns)
    for col in missing_columns:
        df[col] = pd.NA
    return df[expected_columns]

def aggregate_by_date(df):
    if 'DATE' in df.columns:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        df_grouped = df.groupby('DATE', as_index=False).agg({col: 'mean' for col in numeric_cols})
        return df_grouped
    return df

def impute_missing_values(df):
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].mean())
    return df

def process_county_datasets(uploaded_files):
    all_data = {}
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".csv"):
            county_name = os.path.splitext(uploaded_file.name)[0]
            try:
                df = pd.read_csv(uploaded_file)
                df = clean_and_standardize(df)
                df = aggregate_by_date(df)
                df = impute_missing_values(df)
                df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
                all_data[county_name] = df
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
    return all_data

# -------------------------------
# Compatibility Functions
# -------------------------------
def compute_weighted_percentage(row, metric_tuples):
    available_metrics, available_weights = [], []
    for func, col, weight in metric_tuples:
        if col in row and pd.notnull(row[col]):
            val = func(row)
            if val is not None:
                available_metrics.append(val)
                available_weights.append(weight)
    if not available_metrics:
        return 0
    total_weight = sum(available_weights)
    return sum(val * (w / total_weight) for val, w in zip(available_metrics, available_weights))

# -- Wind Energy --
def wind_metric_awnd(row): 
    return min((row['AWND'] / 7) * 100, 100)
def wind_metric_wsf2(row): 
    return min((row['WSF2'] / 15) * 100, 100)
def wind_metric_wsf5(row): 
    return min((row['WSF5'] / 15) * 100, 100)
def wind_metric_elevation(row): 
    return min((row['ELEVATION'] / 500) * 100, 100)
def wind_metric_latlon(row):
    lat = ((row['LATITUDE'] - 33) / 3) * 100
    lon = ((row['LONGITUDE'] - (-94)) / 5) * 100
    return (lat + lon) / 2
def wind_metric_wdf2(row):
    return 100 if pd.notnull(row['WDF2']) else 0
def wind_metric_wdf5(row):
    return 100 if pd.notnull(row['WDF5']) else 0

def compute_wind_compatibility(row):
    metrics = [
        (wind_metric_awnd, 'AWND', 0.40),
        (wind_metric_wsf2, 'WSF2', 0.15),
        (wind_metric_wsf5, 'WSF5', 0.10),
        (wind_metric_latlon, 'LATITUDE', 0.15),
        (wind_metric_elevation, 'ELEVATION', 0.10),
        (wind_metric_wdf2, 'WDF2', 0.05),
        (wind_metric_wdf5, 'WDF5', 0.05)
    ]
    return compute_weighted_percentage(row, metrics)

# -- Solar Energy --
def solar_metric_psun(row): 
    return min((row['PSUN'] / 8) * 100, 100)
def solar_metric_cdsd(row): 
    return min((row['CDSD'] / 24) * 100, 100)
def solar_metric_tmax(row): 
    return min((row['TMAX'] / 90) * 100, 100)
def solar_metric_cldd(row):
    return max(0, 100 - (row['CLDD'] / 50 * 100))
def solar_metric_httd(row):
    return max(0, 100 - (row['HTDD'] / 50 * 100))

def compute_solar_compatibility(row):
    metrics = [
        (solar_metric_psun, 'PSUN', 0.50),
        (solar_metric_cdsd, 'CDSD', 0.20),
        (solar_metric_cldd, 'CLDD', 0.10),
        (solar_metric_tmax, 'TMAX', 0.10),
        (solar_metric_httd, 'HTTD', 0.10)
    ]
    return compute_weighted_percentage(row, metrics)

# -- Biomass Energy --
def biomass_metric_prcp(row): 
    return min((row['PRCP'] / 1500) * 100, 100)
def biomass_metric_tavg(row): 
    return max(0, (1 - abs(row['TAVG'] - 20) / 10)) * 100
def biomass_metric_dp01(row):
    return max(0, 100 - (row['DP01'] / 50 * 100))
def biomass_metric_dx90(row):
    return min((row['DX90'] / 90) * 100, 100)
def biomass_metric_dp10(row):
    return max(0, 100 - (row['DP10'] / 10 * 100))
def biomass_metric_snow(row): 
    return max(0, 100 - (row['SNOW'] / 20 * 100))
def biomass_metric_dx70(row):
    return min((row['DX70'] / 70) * 100, 100)

def compute_biomass_compatibility(row):
    metrics = [
        (biomass_metric_prcp, 'PRCP', 0.20),
        (biomass_metric_tavg, 'TAVG', 0.20),
        (biomass_metric_dp01, 'DP01', 0.15),
        (biomass_metric_dx90, 'DX90', 0.15),
        (biomass_metric_dp10, 'DP10', 0.10),
        (biomass_metric_snow, 'SNOW', 0.10),
        (biomass_metric_dx70, 'DX70', 0.10)
    ]
    return compute_weighted_percentage(row, metrics)

# -- Hydroelectric Energy --
def hydroelectric_metric_prcp(row): 
    return min((row['PRCP'] / 1500) * 100, 100)
def hydroelectric_metric_elevation(row): 
    return min((row['ELEVATION'] / 500) * 100, 100)
def hydroelectric_metric_snow(row): 
    return max(0, 100 - (row['SNOW'] / 20 * 100))
def hydroelectric_metric_tavg(row):
    return max(0, 100 - (abs(row['TAVG'] - 20) / 20 * 100))
def hydroelectric_metric_dt00(row):
    return max(0, 100 - (row['DT00'] / 30 * 100))
def hydroelectric_metric_dt32(row):
    return max(0, 100 - (row['DT32'] / 60 * 100))

def compute_hydroelectric_compatibility(row):
    metrics = [
        (hydroelectric_metric_prcp, 'PRCP', 0.30),
        (hydroelectric_metric_elevation, 'ELEVATION', 0.25),
        (hydroelectric_metric_snow, 'SNOW', 0.10),
        (hydroelectric_metric_tavg, 'TAVG', 0.15),
        (hydroelectric_metric_dt00, 'DT00', 0.10),
        (hydroelectric_metric_dt32, 'DT32', 0.10)
    ]
    return compute_weighted_percentage(row, metrics)

# -------------------------------
# Iterative Forecast Function with Chunk Size = 365 Days and Parallel Processing
# -------------------------------
def iterative_forecast(ts, total_steps, chunk_size=365):
    """Forecast iteratively in chunks (default 365 days per chunk)."""
    forecast_values = []
    current_series = ts.copy()
    steps_remaining = total_steps
    while steps_remaining > 0:
        current_chunk = min(chunk_size, steps_remaining)
        try:
            model = ARIMA(current_series, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit()
            chunk_forecast = model_fit.forecast(steps=current_chunk)
        except Exception as e:
            chunk_forecast = pd.Series([current_series.iloc[-1]] * current_chunk,
                                       index=pd.date_range(start=current_series.index[-1] + pd.Timedelta(days=1), periods=current_chunk))
        forecast_values.extend(chunk_forecast)
        current_series = pd.concat([current_series, chunk_forecast])
        steps_remaining -= current_chunk
    return forecast_values

def compute_actual_compatibility(county_df, energy_type, year):
    df_year = county_df[county_df['DATE'].dt.year == year]
    if df_year.empty:
        return np.nan
    if energy_type == "Wind Compatibility":
        vals = df_year.apply(compute_wind_compatibility, axis=1)
    elif energy_type == "Solar Compatibility":
        vals = df_year.apply(compute_solar_compatibility, axis=1)
    elif energy_type == "Biomass Compatibility":
        vals = df_year.apply(compute_biomass_compatibility, axis=1)
    elif energy_type == "Hydroelectric Compatibility":
        vals = df_year.apply(compute_hydroelectric_compatibility, axis=1)
    return vals.mean() if not vals.empty else np.nan

def compute_forecast_compatibility(county_df, energy_type, forecast_year):
    # Limit training period to 2005-2014
    train_df = county_df[(county_df['DATE'].dt.year >= 2005) & (county_df['DATE'].dt.year <= 2014)]
    if train_df.empty:
        return np.nan
    if energy_type == "Wind Compatibility":
        ts = train_df.apply(compute_wind_compatibility, axis=1)
    elif energy_type == "Solar Compatibility":
        ts = train_df.apply(compute_solar_compatibility, axis=1)
    elif energy_type == "Biomass Compatibility":
        ts = train_df.apply(compute_biomass_compatibility, axis=1)
    elif energy_type == "Hydroelectric Compatibility":
        ts = train_df.apply(compute_hydroelectric_compatibility, axis=1)
    ts.index = train_df['DATE']
    ts = ts.sort_index()
    if ts.std() < 1e-6:
         return ts.iloc[-1]
    years_ahead = forecast_year - 2015
    forecast_days = years_ahead * 365
    try:
        forecast_values = iterative_forecast(ts, forecast_days, chunk_size=365)
        result = forecast_values[-1]
    except Exception as e:
        result = ts.iloc[-1]
    # If result is NaN, fallback to the last non-NaN training value
    if pd.isna(result):
        result = ts.fillna(method='ffill').iloc[-1]
    return result

def parallel_forecast(county_df, energy_type, forecast_year):
    return compute_forecast_compatibility(county_df, energy_type, forecast_year)

# -------------------------------
# Visualization Setup
# -------------------------------
cmap_dict = {
    "Wind Compatibility": "Greys",
    "Solar Compatibility": "YlOrBr",
    "Biomass Compatibility": "Greens",
    "Hydroelectric Compatibility": "Blues"
}

arkansas_counties = [
    "Arkansas", "Ashley", "Baxter", "Benton", "Boone", "Bradley", "Calhoun", "Carroll",
    "Chicot", "Clark", "Clay", "Cleburne", "Cleveland", "Columbia", "Conway", "Craighead",
    "Crawford", "Crittenden", "Cross", "Dallas", "Desha", "Drew", "Faulkner", "Franklin",
    "Fulton", "Garland", "Grant", "Greene", "Hempstead", "Hot Spring", "Howard", "Independence",
    "Izard", "Jackson", "Jefferson", "Johnson", "Lafayette", "Lawrence", "Lee", "Lincoln",
    "Little River", "Logan", "Lonoke", "Madison", "Marion", "Miller", "Mississippi", "Monroe",
    "Montgomery", "Newton", "Nevada", "Ouachita", "Perry", "Phillips", "Pike", "Poinsett", "Polk",
    "Pope", "Prairie", "Pulaski", "Randolph", "Saline", "Scott", "Searcy", "Sebastian", "Sevier",
    "Sharp", "St. Francis", "Stone", "Union", "Van Buren", "Washington", "White", "Woodruff", "Yell"
]

geojson_file = '/Users/juakeensoriano/Downloads/arkansas_counties.geojson'
gdf = gpd.read_file(geojson_file)
gdf = gdf[gdf['STATEFP'] == "05"]
gdf['NAME'] = gdf['NAME'].str.strip().str.title()
arkansas_counties = [c.title() for c in arkansas_counties]
gdf = gdf[gdf['NAME'].isin(arkansas_counties)]
if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)

# -------------------------------
# Streamlit App
# -------------------------------
st.write("### Renewable Energy Compatibility Visualization")

uploaded_files = st.file_uploader("Upload County CSV Files", type="csv", accept_multiple_files=True)

if uploaded_files:
    county_data = process_county_datasets(uploaded_files)
    
    # Actual Compatibility Section
    st.write("#### Actual Compatibility Visualization")
    actual_year = st.slider("Select Actual Year (2005-2025)", min_value=2005, max_value=2025, value=2025, step=1)
    
    actual_results = {etype: {} for etype in cmap_dict.keys()}
    for county, df in county_data.items():
        for etype in cmap_dict.keys():
            actual_results[etype][county] = compute_actual_compatibility(df, etype, actual_year)
    
    # Create a list of counties with non-NaN actual values for the selected energy type
    energy_choice_actual = st.selectbox("Select Energy Type for Actual Compatibility", list(cmap_dict.keys()))
    available_counties_actual = [county for county, val in actual_results[energy_choice_actual].items() if not pd.isna(val)]
    gdf_actual = gdf[gdf['NAME'].isin(available_counties_actual)]
    
    df_actual = pd.DataFrame.from_dict(actual_results[energy_choice_actual], orient='index', columns=[energy_choice_actual])
    df_actual = df_actual.reindex(available_counties_actual)
    gdf_actual = gdf_actual.merge(df_actual, left_on="NAME", right_index=True, how="left")
    
    # Compute relative min and max values for shading
    vmin_actual = gdf_actual[energy_choice_actual].min()
    vmax_actual = gdf_actual[energy_choice_actual].max()
    
    fig_actual, ax_actual = plt.subplots(figsize=(10, 8))
    gdf_actual.plot(column=energy_choice_actual, ax=ax_actual, legend=True,
                    legend_kwds={'label': f'Actual {energy_choice_actual} (%) in {actual_year}',
                                 'orientation': "horizontal"},
                    cmap=cmap_dict.get(energy_choice_actual, "viridis"),
                    edgecolor='k', vmin=vmin_actual, vmax=vmax_actual)
    ax_actual.set_title(f"Actual {energy_choice_actual} ({actual_year})")
    ax_actual.set_axis_off()
    st.pyplot(fig_actual)
    
    actual_df = pd.DataFrame({etype: pd.DataFrame.from_dict(actual_results[etype], orient='index', columns=[etype])[etype]
                              for etype in cmap_dict.keys()})
    actual_df = actual_df.reindex(available_counties_actual)
    st.write(f"### Actual Compatibility Data for {actual_year}")
    st.dataframe(actual_df)
    csv_actual = actual_df.to_csv(index=True).encode('utf-8')
    st.download_button("Download Actual Compatibility Data",
                       data=csv_actual,
                       file_name=f"actual_compatibility_{actual_year}.csv",
                       mime="text/csv")
    
    # Forecasted Compatibility Section
    st.write("#### Forecasted Compatibility Visualization (Validation)")
    forecast_year = st.slider("Select Forecast Year (2016-2025)", min_value=2016, max_value=2025, value=2024, step=1)
    
    forecast_results = {}
    for etype in cmap_dict.keys():
        results = Parallel(n_jobs=8)(
            delayed(parallel_forecast)(df, etype, forecast_year)
            for county, df in county_data.items()
        )
        forecast_results[etype] = {county: val for county, val in zip(county_data.keys(), results)}
    
    # Use only counties that are present in the actual compatibility dataset
    available_counties_forecast = available_counties_actual  
    gdf_forecast = gdf[gdf['NAME'].isin(available_counties_forecast)]
    
    energy_choice_forecast = st.selectbox("Select Energy Type for Forecasted Compatibility", list(cmap_dict.keys()))
    df_forecast = pd.DataFrame.from_dict(forecast_results[energy_choice_forecast], orient='index', columns=[energy_choice_forecast])
    df_forecast = df_forecast.reindex(available_counties_forecast)
    gdf_forecast = gdf_forecast.merge(df_forecast, left_on="NAME", right_index=True, how="left")
    
    # Compute relative min and max for forecast shading
    vmin_forecast = gdf_forecast[energy_choice_forecast].min()
    vmax_forecast = gdf_forecast[energy_choice_forecast].max()
    
    fig_forecast, ax_forecast = plt.subplots(figsize=(10, 8))
    gdf_forecast.plot(column=energy_choice_forecast, ax=ax_forecast, legend=True,
                      legend_kwds={'label': f'Forecast {energy_choice_forecast} (%) in {forecast_year} (trained on 2005-2015)',
                                   'orientation': "horizontal"},
                      cmap=cmap_dict.get(energy_choice_forecast, "viridis"),
                      edgecolor='k', vmin=vmin_forecast, vmax=vmax_forecast)
    ax_forecast.set_title(f"Forecast {energy_choice_forecast} ({forecast_year})")
    ax_forecast.set_axis_off()
    st.pyplot(fig_forecast)
    
    forecast_df = pd.DataFrame({etype: pd.DataFrame.from_dict(forecast_results[etype], orient='index', columns=[etype])[etype]
                                for etype in cmap_dict.keys()})
    forecast_df = forecast_df.reindex(available_counties_forecast)
    st.write(f"### Forecasted Compatibility Data for {forecast_year} (trained on 2005-2015)")
    st.dataframe(forecast_df)
    csv_forecast = forecast_df.to_csv(index=True).encode('utf-8')
    st.download_button("Download Forecasted Compatibility Data",
                       data=csv_forecast,
                       file_name=f"forecasted_compatibility_{forecast_year}.csv",
                       mime="text/csv")
    
else:
    st.warning("Please upload at least one county CSV file.")
