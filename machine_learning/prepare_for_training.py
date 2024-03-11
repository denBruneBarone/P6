import torch
from torch.utils.data import Dataset
from data_processing.energy_consumption.trapeziod_integration import integrate_flight_data
import pandas as pd

class TrainingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        # Extract features
        payload = float(sample['payload'])
        speed = float(sample['speed'])
        flight = sample['flight']

        # TODO: Probably do normalization before this!!!
        sequential_data = sample['data']  # Assuming this is a DataFrame

        # TODO: undersøg om vi vil bruge Time-Series Feature Extraction eller Temporal Aggregation osv. OBS: nok ikke, vi bruger LSTM.

        # statisk input
        input_tensor = torch.tensor([payload, speed], dtype=torch.float)

        # sequentielt input
        sequential_tensor = torch.tensor(sequential_data.values, dtype=torch.float)

        # outout/target feature
        target_tensor = torch.tensor(float(sample['power']),
                                     dtype=torch.float)

        return input_tensor, sequential_tensor, target_tensor


# Takes df parameter
def organize_data(array_of_df):
    flight_dict_list = []

    for df in array_of_df:
        df['position_x'] = df['position_x'] - df['position_x'].iloc[0]
        df['position_y'] = df['position_y'] - df['position_y'].iloc[0]
        df['position_z'] = df['position_z'] - df['position_z'].iloc[0]

        flight = df['flight'].iloc[0]
        payload = df['payload'].iloc[0]
        # speed = df['speed'].iloc[0]
        df = df.drop(columns=['flight', 'speed', 'payload', 'altitude', 'date', 'time_day', 'route'])

        flight_dict = {
            "flight": flight,
            "data": df,
            "payload": payload,
            # "speed": speed,
            "power": integrate_flight_data(df)
        }
        flight_dict_list.append(flight_dict)
        print(flight_dict)
    return flight_dict_list
