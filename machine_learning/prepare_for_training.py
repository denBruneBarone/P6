from data_processing.energy_consumption.datapoints_summation import add_power_to_df
from sklearn.preprocessing import StandardScaler
import pandas as pd
import math


class TrainingDataset:
    def __init__(self, data):
        self.data = data

    # Kigger på alt given data, ikke bare en enkelt dataframe. Konstruerer en passende scaler.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        # input features
        input_array = sample[
            ['time', 'wind_speed', 'wind_angle',
             'position_x', 'position_y', 'position_z',
             'velocity_x', 'velocity_y', 'velocity_z',
             'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
             'payload']
        ].values


        # Output/target feature
        target_array = sample[
            ['battery_current', 'battery_voltage']
        ].values

        return input_array, target_array


def format_data(input_data):
    if isinstance(input_data, list):
        formatted_array = []
        for df in input_data:
            start_diff_lon = df['position_x'] - df['position_x'].iloc[0]
            start_diff_lat = df['position_y'] - df['position_y'].iloc[0]
            df['position_x'], df['position_y'] = calculate_lat_lon_distance(start_diff_lon, start_diff_lat)

            df['position_z'] = df['position_z'] - df['position_z'].iloc[0]

            df['battery_current'] = df['battery_current'].apply(lambda x: max(x, 0))

            df = df.drop(columns=['flight', 'speed', 'altitude', 'date', 'time_day', 'route'])
            formatted_array.append(df)
        return formatted_array
    elif isinstance(input_data, pd.DataFrame):
        df = input_data
        start_diff_lon = df['position_x'] - df['position_x'].iloc[0]
        start_diff_lat = df['position_y'] - df['position_y'].iloc[0]
        df['position_x'], df['position_y'] = calculate_lat_lon_distance(start_diff_lon, start_diff_lat)

        df['position_z'] = df['position_z'] - df['position_z'].iloc[0]

        df['battery_current'] = df['battery_current'].apply(lambda x: max(x, 0))

        df = df.drop(columns=['flight', 'speed', 'altitude', 'date', 'time_day', 'route'])
        return df


def calculate_lat_lon_distance(delta_lat, delta_lon):
    # Earth's radius in meters along one degree of latitude (approximate)
    lat = -79.78239570000002
    meters_per_degree = 111320

    # Calculate distance for the latitude
    distance_lat = delta_lat * meters_per_degree

    # Calculate distance for the longitude
    # Convert latitude to radians for the cosine function
    lat_radians = math.radians(lat)
    distance_lon = delta_lon * meters_per_degree * math.cos(lat_radians)

    return distance_lat, distance_lon


