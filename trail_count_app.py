import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import pickle

# Generic Austin Lat/Long
AUSTIN_LAT = 30.28
AUSTIN_LONG = -97.75

# Page config for better visuals
st.set_page_config(
    page_title="Austin Daily Trail Count Forecast", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .download-btn {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-align: center;
        display: inline-block;
        text-decoration: none;
    }
    .download-btn:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize in-memory cache
# Key: forecast start date (str), Value: (DataFrame, date_range)
forecast_cache = {}

# Load models, scalers, and trail locations
models = pickle.load(open('trail_count_models.pkl', 'rb'))
scalers = pickle.load(open('scalers.pkl', 'rb'))
trail_locations = pd.read_csv('trail_locations.csv')

# Load historical data for lag1_count and histograms
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
    fig = px.histogram(
        x=counts,
        nbins=20,
        title="",
        height=100,
        width=250,
        range_x=[counts.min(), counts.max()]
    )
    count_range = counts.max() - counts.min()
    mid_point = counts.min() + (count_range / 2)
    annotation_pos = "top left" if predicted_count > mid_point else "top right"

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
    padding = count_range * 0.05
    fig.update_xaxes(range=[counts.min() - padding, counts.max() + padding])

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

# Function to determine weather icon based on conditions


def get_weather_icon(precipitation, cloud_cover):
    if precipitation > 0.1:
        return "🌧️"  # Rain
    elif cloud_cover > 70:
        return "☁️"  # Cloudy
    elif cloud_cover > 30:
        return "⛅"  # Partly cloudy
    else:
        return "☀️"  # Sunny

# Function to get trend arrow based on change


def get_trend_arrow(change):
    if change > 0:
        return "⬆️"  # Up arrow
    elif change < 0:
        return "⬇️"  # Down arrow
    else:
        return "➡️"  # Neutral arrow

# Function to fetch Open Meteo forecast with date-based caching


def get_open_meteo_forecast(selected_date_str, lat=AUSTIN_LAT, lon=AUSTIN_LONG):
    selected_date = pd.to_datetime(selected_date_str)
    forecast_start = datetime.today().date()
    forecast_start_str = forecast_start.strftime('%Y-%m-%d')

    for cached_start, (cached_df, date_range) in forecast_cache.items():
        if selected_date.date() in date_range:
            return cached_df, f"Using cached forecast from {cached_start}"

    with st.spinner("Fetching weather forecast..."):
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={AUSTIN_LAT}"
               f"&longitude={AUSTIN_LONG}&daily=apparent_temperature_max,"
               f"apparent_temperature_min,wind_speed_10m_max,precipitation_sum,"
               f"cloud_cover_mean&hourly=temperature_2m&timezone=auto&"
               f"wind_speed_unit=mph&temperature_unit=fahrenheit&precipitation_unit=inch")

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

            date_range = pd.date_range(
                forecast_df['Date'].min(), forecast_df['Date'].max())
            date_range = set(date_range.date)

            forecast_cache[forecast_start_str] = (forecast_df, date_range)

            if len(forecast_cache) > 1:
                oldest_key = min(forecast_cache.keys())
                del forecast_cache[oldest_key]

            return forecast_df, f"Fetched new forecast for {forecast_start_str}"
        except requests.RequestException as e:
            if forecast_cache:
                st.warning(
                    f"Failed to fetch forecast: {e}. Using last cached forecast.")
                return forecast_cache[max(forecast_cache.keys())][0], f"Using last cached forecast from {max(forecast_cache.keys())}"
            st.error(f"Failed to fetch forecast: {e}")
            return pd.DataFrame(), "Failed to fetch forecast."


# Streamlit app
st.title("Austin Daily Trail Count Forecast")
st.markdown("*(pedestrians and bicyclists)*")
st.divider()

# Date selection
today = datetime.today()
date_options = [(today + timedelta(days=i)).strftime('%Y-%m-%d')
                for i in range(7)]
selected_date = st.selectbox("Select Forecast Date", date_options)

# Fetch forecast data
forecast_df, cache_message = get_open_meteo_forecast(selected_date)
st.write(cache_message)

if not forecast_df.empty:
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.date
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    forecast_data = forecast_df[forecast_df['Date'].dt.strftime(
        '%Y-%m-%d') == selected_date]

    if not forecast_data.empty:
        # Display weather info with icons
        st.subheader(f"Weather Forecast for :blue-background[{selected_date}]")
        weather_icon = get_weather_icon(forecast_data['precipitation_sum (inch)'].iloc[0],
                                        forecast_data['cloud_cover_mean (%)'].iloc[0])
        st.markdown(f"**Day's Condition: {weather_icon}**")

        cols = st.columns([1, 1, 1, 1])
        cols[0].write("**Max Feels Like Temp:**")
        cols[1].write(
            f"{forecast_data['apparent_temperature_max (°F)'].iloc[0]:.1f}°F")
        cols[2].write("**Precipitation:**")
        cols[3].write(
            f"{forecast_data['precipitation_sum (inch)'].iloc[0]:.3f} in")

        cols = st.columns([1, 1, 1, 1])
        cols[0].write("**Min Feels Like Temp:**")
        cols[1].write(
            f"{forecast_data['apparent_temperature_min (°F)'].iloc[0]:.1f}°F")
        cols[2].write("**Cloud Cover:**")
        cols[3].write(f"{forecast_data['cloud_cover_mean (%)'].iloc[0]:.1f}%")

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

            hist_fig = create_histogram(sensor_id, pred_count)

            # Calculate change from average
            avg_count = avg_counts.get(sensor_id, 200)
            change = ((pred_count - avg_count) / avg_count *
                      100) if avg_count != 0 else 0
            trend_arrow = get_trend_arrow(change)

            predictions.append({
                'Name': loc['Name'],
                'Latitude': loc['Latitude'],
                'Longitude': loc['Longitude'],
                'Predicted Count': max(0, pred_count),
                'Change from Avg': f"{change:.1f}% {trend_arrow}",
                'Histogram': hist_fig
            })

        pred_df = pd.DataFrame(predictions)

        # Create map with single-color markers
        st.subheader("Predicted Trail Usage Map")
        m = folium.Map(location=[AUSTIN_LAT, AUSTIN_LONG], zoom_start=11)
        for _, row in pred_df.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=row['Predicted Count'] / 50,
                popup=f"{row['Name']}: {int(row['Predicted Count'])} users",
                color='blue',
                fill=True,
                fill_color='blue'
            ).add_to(m)

        st_folium(m, width=700, height=500)

        # Display predictions table with histograms and change from avg
        st.subheader("Predicted Trail Counts")
        cols = st.columns([2, 1, 1, 2])
        cols[0].write("**Trail Name**")
        cols[1].write("**Predicted Count**")
        cols[2].write("**Change from Avg**")
        cols[3].write("**Historical Distribution**")

        for _, row in pred_df.iterrows():
            cols = st.columns([2, 1, 1, 2])
            cols[0].write(row['Name'])
            cols[1].write(int(row['Predicted Count']))
            cols[2].write(row['Change from Avg'])
            if row['Histogram'] is not None:
                cols[3].plotly_chart(
                    row['Histogram'], use_container_width=True)
            else:
                cols[3].write("No historical data")
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

Open Meteo [Weather Data](https://open-meteo.com/)

Austin [Trail Counters Device Locations](https://data.austintexas.gov/Transportation-and-Mobility/Trail-Counters-Device-Locations/vxcr-pjs7/about_data)

Austin [Trail Counters Daily Totals](https://data.austintexas.gov/Transportation-and-Mobility/Trail-Counters-Daily-Totals/26tt-cp67/about_data) dataset

> This is an internal dataset that shows the twenty-four hour counts from automated trail counter stations. It gives a count of pedestrians and bicyclists that passed by the device location sensor.   
""")
