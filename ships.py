# Get ship statistics along interpolated best deck, using max distance from interpolated center
# Max distance
max_distance = 500  # km

bdeck_url = 'https://ftp.nhc.noaa.gov/atcf/btk/bal992024.dat'
# alternative source
#bdeck_url = 'https://hurricanes.ral.ucar.edu/repository/data/bdecks_open/2024/bal992024.dat'

ships_url = 'https://www.ndbc.noaa.gov/ship_obs.php'
# Last 12 hours
params = {'uom': 'M', 'time': '12'}

# example to find times nearest buoy
buoy_coord = [34.703, -72.242]

from datetime import datetime, timedelta
import sys

# Get the current year
year = datetime.now().strftime('%Y')

# Check if there are any command-line arguments
if len(sys.argv) > 1:
    arg = sys.argv[1].lower()

    # Check if the argument matches the pattern (AL99, AL05, etc.)
    if len(arg) >= 4 and arg[:2].isalpha() and arg[2:].isdigit():
        basin = arg[:2]
        number = arg[2:]

        # Modify the URL
        bdeck_url = f'https://ftp.nhc.noaa.gov/atcf/btk/b{basin}{number}{year}.dat'

import matplotlib.dates as mdates
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#from geopy.interpolate import interpolate
# geopy 2.0
from geographiclib.geodesic import Geodesic
import os

import requests

def datetime_format(dt):
    return dt.strftime("%m/%d %HZ")

# Define the interpolation function
def interpolate_coordinates(row, next_row, fraction):
    lat1, lon1 = row['lat'], row['lon']
    lat2, lon2 = next_row['lat'], next_row['lon']
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)
    g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    distance = g['s12']
    interpolated_distance = distance * fraction
    interpolated_line = Geodesic.WGS84.Direct(lon1=lon1, lat1=lat1, azi1=g['azi1'], s12=interpolated_distance)
    lat2 = interpolated_line['lat2']
    lon2 = interpolated_line['lon2']
    return pd.Series({'lat': lat2, 'lon': lon2})

# Request data
response = requests.get(ships_url, params=params)

# Parse HTML
soup = BeautifulSoup(response.text, 'html.parser')

# Find the observation count paragraph
obs_count_p = soup.find('p', class_='obs-count')

# Extract the datetime string
dt_str = obs_count_p.text.split('to')[-1].strip()

# Parse the datetime string
end_dt = datetime.strptime(dt_str, '%m/%d/%Y %H%M GMT')

# Find all observation periods
obs_periods = soup.find_all('pre', class_='wide-content')

# Extract data
data = []
for period in obs_periods:
    # Find all lines in the observation period
    lines = period.find_all('span')

    # Skip the header line
    for line in lines[1:]:
        # Split the line into columns
        cols = line.text.strip().split()

        # Replace '-' with NaN
        cols = ['NaN' if col == '-' else col for col in cols]

        # Extract the data
        hour = cols[1]
        lat = float(cols[2]) if cols[2] != 'NaN' else np.nan
        lon = float(cols[3]) if cols[3] != 'NaN' else np.nan
        pres = float(cols[9]) if cols[9] != 'NaN' else np.nan
        wspd = float(cols[5]) if cols[5] != 'NaN' else np.nan

        # invalid
        if wspd > 200:
            continue

        # Append the data to the list
        data.append([hour, lat, lon, pres, wspd])

# Convert data to numeric
data = np.array(data, dtype=float)

# Convert data to numeric
data_numeric = np.array(data, dtype=float)

# Create a separate array for the datetimes
data_datetimes = np.empty(len(data), dtype=object)

# Convert the hour fields to datetimes
for i in range(len(data)):
    hour = int(data[i, 0])
    if hour > end_dt.hour:
        data_datetimes[i] = (end_dt - timedelta(days=1)).replace(hour=hour, minute=0, second=0, microsecond=0)
    else:
        data_datetimes[i] = end_dt.replace(hour=hour, minute=0, second=0, microsecond=0)

# Create a new array with the datetime array and the rest of the numeric data
data_new = np.column_stack((data_datetimes, data_numeric[:, 1:]))

# Replace the original data with the new data
data_ships = data_new

# Read the .dat file from the web
response = requests.get(bdeck_url)
lines = response.text.splitlines()

# Parse the .dat file to get a df of the valid_time, lon, and lat
data = []
for line in lines:
    cols = line.split(',')
    valid_time = pd.to_datetime(cols[2].strip(), format='%Y%m%d%H')
    lat_dir = cols[6][-1]
    lat = float(cols[6][:-1].strip()) / 10
    if lat_dir == 'S':
        lat = -lat
    lon_dir = cols[7][-1]
    lon = float(cols[7][:-1].strip()) / 10
    if lon_dir == 'W':
        lon = -lon
    data.append([valid_time, lon, lat])

df = pd.DataFrame(data, columns=['valid_time', 'lon', 'lat'])

# Sort the DataFrame by valid_time
df.sort_values(by='valid_time', inplace=True)

df.reset_index(inplace=True)

# Interpolate the coordinates at each hour
for i in range(len(df) - 1):
    row = df.iloc[i]
    next_row = df.iloc[i + 1]
    for j in range(1, 6):  # interpolate at each hour
        fraction = j / 6
        interpolated_row = interpolate_coordinates(row, next_row, fraction)
        # Create a new datetime for the interpolated row
        new_datetime = row['valid_time'] + pd.Timedelta(hours=j)
        interpolated_row['valid_time'] = new_datetime
        df = pd.concat([df, pd.DataFrame([interpolated_row])], ignore_index=True)

# Sort the DataFrame by valid_time (needed again after interpolation)
df.sort_values(by='valid_time', inplace=True)

df.reset_index(inplace=True)

buoy_dists = []
# buoy example
for i, row in df.iterrows():
    g = Geodesic.WGS84.Inverse(buoy_coord[0], buoy_coord[1], row.lat, row.lon)
    buoy_dists.append((np.round(g['s12'] / 1000, 1), np.round(g['azi1'],1), row['valid_time']))

buoy_dists.sort()
buoy_df = pd.DataFrame(buoy_dists, columns=['distance_km', 'azimuth', 'valid_time'])

print(f"Closest storm approaches to buoy at {buoy_coord[0], buoy_coord[1]}:")
print(buoy_df)

print("")
print('Interpolated best track:')
print(df)

distances = []
for i in range(len(data_ships)):
    # Find the row for the valid_time that is closest to the datetime in the data
    closest_row = df.iloc[(df['valid_time'] - data_ships[i, 0]).abs().argsort()[:1]]
    center_lat = closest_row['lat'].values[0]
    center_lon = closest_row['lon'].values[0]
    # Calculate the distance based on the center
    g = Geodesic.WGS84.Inverse(center_lat, center_lon, data_ships[i, 1], data_ships[i, 2])
    distance_km = g['s12'] / 1000
    # Use the distance in your calculations...
    distances.append(distance_km)

distances = np.array(distances)

# Remove data points where distance is greater than max_distance
mask = distances <= max_distance
data_ships = data_ships[mask]
distances = np.array(distances)[mask]

valid_time = data_ships[:, 0]
# Extract pressures and wind speeds
pressures = data_ships[:, 3]
wind_speeds = data_ships[:, 4]

# Create a new array with an additional column for distances
data_ships_with_distances = np.hstack((data_ships, distances.reshape(-1, 1)))

# Convert the resulting array to a pandas DataFrame
df_ships = pd.DataFrame(data_ships_with_distances, columns=['datetime', 'lon', 'lat', 'pressure', 'speed', 'distance'])

print("")
print("Ships observations")
print(df_ships)

fig = plt.figure(figsize=(18, 8))

axs = [
    plt.subplot2grid((2, 2), (0, 0)),
    plt.subplot2grid((2, 2), (0, 1)),
    plt.subplot2grid((2, 2), (1, 0)),
    plt.subplot2grid((2, 2), (1, 1)),
]

axs[0].scatter(valid_time, pressures)
axs[0].set_title('Pressure (mbar) over Time')
axs[0].set_xlabel('Time (UTC)')
axs[0].set_ylabel('Value')
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

axs[1].scatter(valid_time, wind_speeds)
axs[1].set_title('Wind Speed (m/s) over Time')
axs[1].set_xlabel('Time (UTC)')
axs[1].set_ylabel('Value')
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

axs[2].scatter(distances, pressures)
axs[2].set_title(f'Pressure (mbar) over Distance\nMin Pressure: {np.round(np.nanmin(pressures), 1)} mbar at {np.round(distances[np.nanargmin(pressures)], 1)} km, {datetime_format(df_ships.iloc[np.nanargmin(pressures)]["datetime"])}')
axs[2].set_xlabel('Distance (km)')
axs[2].set_ylabel('Value')

axs[3].scatter(distances, wind_speeds)
axs[3].set_title(f'Wind Speed (m/s) over Distance\nMax Speed: {np.round(np.nanmax(wind_speeds), 1)} m/s at {np.round(distances[np.nanargmax(wind_speeds)], 1)} km, {datetime_format(df_ships.iloc[np.nanargmax(wind_speeds)]["datetime"])}')
axs[3].set_xlabel('Distance (km)')
axs[3].set_ylabel('Value')

# Get the filename without the extension
filename = os.path.splitext(os.path.basename(bdeck_url))[0]

plt.suptitle(f'NDBC Ships Reports along (Interpolated) best track (NHC) from {filename}')

plt.tight_layout()

# Save the figure to a PNG file
plt.savefig(f'ships_reports_{filename}.png', bbox_inches='tight', dpi=300)

plt.show()
