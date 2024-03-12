import torch
from torch.utils.data import Dataset
from data_processing.energy_consumption.trapeziod_integration import integrate_flight_data
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


class TrainingDataset(Dataset):
    def __init__(self, data, max_seq_length):
        self.data = data
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        # Extract features
        payload = float(sample['payload'])
        # TODO: Probably do normalization before this!!!
        sequential_data = sample['data']  # Assuming this is a DataFrame

        # Pad sequential_data to max_seq_length
        original_length = len(sequential_data)
        padded_sequence = self.pad_sequence(sequential_data, self.max_seq_length)

        # statisk input
        input_tensor = torch.tensor([payload], dtype=torch.float)

        # sequentielt input
        sequential_tensor = torch.tensor(padded_sequence.values, dtype=torch.float)

        # outout/target feature
        target_tensor = torch.tensor(float(sample['power']), dtype=torch.float)

        original_length_tensor = torch.tensor(original_length, dtype=torch.long)

        return input_tensor, sequential_tensor, target_tensor, original_length_tensor

    def pad_sequence(self, sequence, max_seq_length):
        # Pad the sequence up to max_seq_length
        if len(sequence) < max_seq_length:
            padding_size = max_seq_length - len(sequence)
            padding = pd.DataFrame([[0] * len(sequence.columns)] * padding_size, columns=sequence.columns)
            sequence = pd.concat([sequence, padding], ignore_index=True)
        elif len(sequence) > max_seq_length:
            sequence = sequence.iloc[:max_seq_length, :]

        return sequence



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
    return flight_dict_list
