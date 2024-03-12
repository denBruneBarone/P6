import numpy as np
from torch.utils.data import Dataset
from data_processing.energy_consumption.trapeziod_integration import add_power_to_df
from sklearn.preprocessing import StandardScaler
import pandas as pd


class TrainingDataset:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.fit_scaler()  # Fit the scaler on initialization

    def fit_scaler(self):
        # Concatenate all DataFrames in self.data into a single DataFrame
        df = pd.concat(self.data, ignore_index=True)

        features = df[
            ['time', 'wind_speed', 'wind_angle',
             'battery_voltage', 'battery_current',
             'position_x', 'position_y', 'position_z',
             'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
             'velocity_x', 'velocity_y', 'velocity_z',
             'angular_x', 'angular_y', 'angular_z',
             'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
             'payload']
        ].values

        self.scaler.fit(features)  # Fit the scaler on the entire training dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        # input features
        input_array = sample[
            ['time', 'wind_speed', 'wind_angle',
             'battery_voltage', 'battery_current',
             'position_x', 'position_y', 'position_z',
             'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
             'velocity_x', 'velocity_y', 'velocity_z',
             'angular_x', 'angular_y', 'angular_z',
             'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
             'payload']
        ].values

        # Normalize input features using the previously fitted scaler
        normalized_input = self.scaler.transform(input_array)

        # Output/target feature
        target_array = sample['cumulative_power'].values.reshape(-1, 1)

        return normalized_input, target_array


def format_data(array_of_df):
    formatted_array = []
    for df in array_of_df:
        df['position_x'] = df['position_x'] - df['position_x'].iloc[0]
        df['position_y'] = df['position_y'] - df['position_y'].iloc[0]
        df['position_z'] = df['position_z'] - df['position_z'].iloc[0]

        df = df.drop(columns=['flight', 'speed', 'altitude', 'date', 'time_day', 'route'])
        df = add_power_to_df(df)
        formatted_array.append(df)
    return formatted_array
