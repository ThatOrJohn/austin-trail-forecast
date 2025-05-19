import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Generic Austin Lat/Long
AUSTIN_LAT = 30.28
AUSTIN_LONG = -97.75

# Initialize in-memory cache
# Key: forecast start date (str), Value: (DataFrame, date_range)
forecast_cache = {}

# Load models, scalers, and trail locations
models = pickle.load(open('trail_count_models.pkl', 'rb'))
scalers = pickle.load(open('scalers.pkl', 'rb'))
trail_locations = pd.read_csv('trail_locations.csv')

# Load historical data for lag1_count
trail_counts = pd.read_csv('Trail_Counters_Daily_Totals_20250518.csv',
                           parse_dates=['Date'],
                           date_format='%m/%d/%Y %I:%M:%S %p')
trail_counts = trail_counts[['Date', 'Sensor ID', 'Count']]
avg_counts = trail_counts.groupby('Sensor ID')['Count'].mean().to_dict()

# Precompute historical count distributions for histograms
histogram_data = {}
for sensor_id in trail_locations['Sensor ID']:
    sensor_counts = trail_counts[trail_counts['Sensor ID']
                                 == sensor_id]['Count']
    if not sensor_counts.empty:
        # Define bins (20 bins from min to max count)
        hist, bins = np.histogram(sensor_counts, bins=20, range=(
            sensor_counts.min(), sensor_counts.max()))
        histogram_data[sensor_id] = {
            'counts': sensor_counts,
            'hist': hist,
            'bins': bins
        }

# Function to create a histogram with predicted count marker


def create_histogram(sensor_id, predicted_count):
    if sensor_id not in histogram_data:
        return None

    counts = histogram_data[sensor_id]['counts']

    # Create histogram with Plotly
    fig = px.histogram(
        x=counts,
        nbins=20,
        title="",
        height=100,
        width=250,
        range_x=[counts.min(), counts.max()]
    )

    # Determine annotation position based on predicted count's position
    count_range = counts.max() - counts.min()
    mid_point = counts.min() + (count_range / 2)
    annotation_pos = "top left" if predicted_count > mid_point else "top right"

    # Add vertical line for predicted count
    fig.add_vline(
        x=predicted_count,
        line_dash="dash",
        line_color="red",
        annotation_text=f"{int(predicted_count)}",
        annotation_position=annotation_pos,
        annotation_textangle=0,
        annotation_font_size=10,
        annotation_font_color="white",
        annotation_bgcolor="rgba(0, 0, 0, 0.5)"
    )

    # Extend x-axis range slightly to prevent clipping
    padding = count_range * 0.05  # 5% padding on each side
    fig.update_xaxes(range=[counts.min() - padding, counts.max() + padding])

    # Style for compact display
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="",
        yaxis_title="",
        xaxis_tickfont_size=10,
        yaxis_tickfont_size=10,
        bargap=0.1
    )
    return fig

# Function to fetch Open Meteo forecast with date-based caching


def get_open_meteo_forecast(selected_date_str, lat=AUSTIN_LAT, lon=AUSTIN_LONG):
    # Convert selected date to datetime
    selected_date = pd.to_datetime(selected_date_str)

    # Forecast start date is today (or the earliest date in the 7-day range)
    forecast_start = datetime.today().date()
    forecast_start_str = forecast_start.strftime('%Y-%m-%d')

    # Check if cached forecast covers the selected date
    for cached_start, (cached_df, date_range) in forecast_cache.items():
        if selected_date.date() in date_range:
            st.write(f"Using cached forecast from {cached_start}")
            return cached_df

    # Fetch new forecast
    url = f"https://api.open-meteo.com/v1/forecast?latitude={AUSTIN_LAT}&longitude={AUSTIN_LONG}&daily=apparent_temperature_max,apparent_temperature_min,wind_speed_10m_max,precipitation_sum,cloud_cover_mean&hourly=temperature_2m&timezone=auto&wind_speed_unit=mph&temperature_unit=fahrenheit&precipitation_unit=inch"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        forecast = []
        for i in range(len(data['daily']['time'])):
            forecast.append({
                'Date': data['daily']['time'][i],
                'apparent_temperature_max (°F)': data['daily']['apparent_temperature_max'][i],
                'apparent_temperature_min (°F)': data['daily']['apparent_temperature_min'][i],
                'wind_speed_10m_max (mp/h)': data['daily']['wind_speed_10m_max'][i],
                'precipitation_sum (inch)': data['daily']['precipitation_sum'][i],
                'cloud_cover_mean (%)': data['daily']['cloud_cover_mean'][i]
            })
        forecast_df = pd.DataFrame(forecast)
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

        # Create date range for cache
        date_range = pd.date_range(
            forecast_df['Date'].min(), forecast_df['Date'].max())
        date_range = set(date_range.date)  # Convert to set of dates

        # Store in cache
        forecast_cache[forecast_start_str] = (forecast_df, date_range)
        st.write(f"Fetched new forecast for {forecast_start_str}")

        # Clean old cache entries (keep only the latest)
        if len(forecast_cache) > 1:
            oldest_key = min(forecast_cache.keys())
            del forecast_cache[oldest_key]

        return forecast_df
    except requests.RequestException as e:
        st.error(f"Failed to fetch forecast: {e}")
        # Fallback to most recent cached forecast
        if forecast_cache:
            st.warning(
                f"Using last cached forecast from {max(forecast_cache.keys())}")
            return forecast_cache[max(forecast_cache.keys())][0]
        return pd.DataFrame()


# Streamlit app
st.title("Austin Daily Trail Count Forecast")
st.markdown("*(pedestrians and bicyclists)*")
st.divider()

# Date selection
today = datetime.today()
date_options = [(today + timedelta(days=i)).strftime('%Y-%m-%d')
                for i in range(7)]
selected_date = st.selectbox("**Select Forecast Date**", date_options)

# Fetch forecast data
forecast_df = get_open_meteo_forecast(selected_date)
if not forecast_df.empty:
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.date
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    forecast_data = forecast_df[forecast_df['Date'].dt.strftime(
        '%Y-%m-%d') == selected_date]

    if not forecast_data.empty:
        max_temp = forecast_data['apparent_temperature_max (°F)'].iloc[0]
        precipitation = forecast_data['precipitation_sum (inch)'].iloc[0]
        # Easter egg
        if max_temp < 33 and precipitation > 0:
            st.snow()
        # Display weather info as a paired table
        st.subheader(f"Weather Forecast for :blue-background[{selected_date}]")

        # Row 1: Max Feels Like Temp and Precipitation
        cols = st.columns([1, 1, 1, 1])
        cols[0].write("**Max Feels Like Temp:**")
        cols[1].write(
            f"{max_temp:.1f}°F")
        cols[2].write("**Precipitation:**")
        cols[3].write(
            f"{precipitation:.3f} in")

        # Row 2: Min Feels Like Temp and Cloud Cover
        cols = st.columns([1, 1, 1, 1])
        cols[0].write("**Min Feels Like Temp:**")
        cols[1].write(
            f"{forecast_data['apparent_temperature_min (°F)'].iloc[0]:.1f}°F")
        cols[2].write("**Cloud Cover:**")
        cols[3].write(f"{forecast_data['cloud_cover_mean (%)'].iloc[0]:.1f}%")

        # Row 3: Wind Speed and empty pair
        cols = st.columns([1, 1, 1, 1])
        cols[0].write("**Wind Speed:**")
        cols[1].write(
            f"{forecast_data['wind_speed_10m_max (mp/h)'].iloc[0]:.1f} mph")
        cols[2].write("")
        cols[3].write("")

        # Prepare predictions
        predictions = []
        selected_date_dt = pd.to_datetime(selected_date)

        for _, loc in trail_locations.iterrows():
            sensor_id = loc['Sensor ID']
            if sensor_id not in models:
                continue

            lag1_count = avg_counts.get(sensor_id, 200)
            lag1_precipitation = 0.0

            input_data = pd.DataFrame({
                'apparent_temperature_max (°F)': [forecast_data['apparent_temperature_max (°F)'].iloc[0]],
                'apparent_temperature_min (°F)': [forecast_data['apparent_temperature_min (°F)'].iloc[0]],
                'wind_speed_10m_max (mp/h)': [forecast_data['wind_speed_10m_max (mp/h)'].iloc[0]],
                'precipitation_sum (inch)': [forecast_data['precipitation_sum (inch)'].iloc[0]],
                'cloud_cover_mean (%)': [forecast_data['cloud_cover_mean (%)'].iloc[0]],
                'Day_of_Week': [selected_date_dt.dayofweek],
                'Is_Weekend': [1 if selected_date_dt.dayofweek in [5, 6] else 0],
                'Month': [selected_date_dt.month],
                'Temp_Range': [forecast_data['apparent_temperature_max (°F)'].iloc[0] -
                               forecast_data['apparent_temperature_min (°F)'].iloc[0]],
                'lag1_count': [lag1_count],
                'lag1_precipitation': [lag1_precipitation]
            })

            scaler = scalers[sensor_id]
            input_scaled = scaler.transform(input_data)
            model = models[sensor_id]
            pred_log_count = model.predict(input_scaled)[0]
            pred_count = np.expm1(pred_log_count)

            # Create histogram for this prediction
            hist_fig = create_histogram(sensor_id, pred_count)

            predictions.append({
                'Name': loc['Name'],
                'Latitude': loc['Latitude'],
                'Longitude': loc['Longitude'],
                'Predicted_Count': max(0, pred_count),
                'Histogram': hist_fig
            })

        pred_df = pd.DataFrame(predictions)

        st.divider()

        # Create map
        st.subheader("Predicted Trail Usage Map")
        m = folium.Map(location=[30.2672, -97.7431], zoom_start=11)
        for _, row in pred_df.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=row['Predicted_Count'] / 75,
                popup=f"{row['Name']}: {int(row['Predicted_Count'])} users",
                color='blue',
                fill=True,
                fill_color='blue'
            ).add_to(m)
        st_folium(m, width=700, height=500)

        # Display predictions table with histograms
        st.subheader("Predicted Trail Counts")
        # Create columns for table display
        # Adjust widths: Name, Predicted_Count, Histogram
        cols = st.columns([2, 1, 2])
        cols[0].write("**Trail Name**")
        cols[1].write("**Predicted Count**")
        cols[2].write("**Historical Distribution**")

        for _, row in pred_df.iterrows():
            cols = st.columns([2, 1, 2])
            cols[0].write(row['Name'])
            cols[1].write(int(row['Predicted_Count']))
            with st.container():
                if row['Histogram'] is not None:
                    cols[2].plotly_chart(
                        row['Histogram'], use_container_width=True)
                else:
                    cols[2].write("No historical data")
    else:
        st.error("No forecast data available for the selected date.")
else:
    st.error("Failed to retrieve forecast data.")

st.divider()
st.subheader("Background")
st.markdown("""
This app predicts daily trail usage for three Austin trails using XGBoost models trained on weather and historical count data. 
Key features include temperature, precipitation, cloud cover, and day-of-week effects. 
R² scores range from 0.67 to 0.72, indicating good predictive accuracy.         
            

Based on these datasets:
            
Open  Meteo [Weather Data](https://open-meteo.com/)
            
Austin [Trail Counters Device Locations](https://data.austintexas.gov/Transportation-and-Mobility/Trail-Counters-Device-Locations/vxcr-pjs7/about_data)

Austin [Trail Counters Daily Totals](https://data.austintexas.gov/Transportation-and-Mobility/Trail-Counters-Daily-Totals/26tt-cp67/about_data) dataset

> This is an internal dataset that shows the twenty-four hour counts from automated trail counter stations. It gives a count of pedestrians and bicyclists that passed by the device location sensor.   
""")
