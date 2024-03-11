import torch
from torch.utils.data import Dataset
from data_processing.energy_consumption.trapeziod_integration import integrate_flight_data

class TrainingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index].values
        sample = torch.tensor(sample, dtype=torch.float)
        return sample


# Takes df parameter
def organize_data(array_of_df):
    flight_dict_list = []

    for df in array_of_df:
        df['position_x'] = df['position_x'] - df['position_x'].iloc[0]
        df['position_y'] = df['position_y'] - df['position_y'].iloc[0]
        df['position_z'] = df['position_z'] - df['position_z'].iloc[0]

        flight_dict = {
            "flight": df['flight'].iloc[0],
            "data": df,
            "payload": df['payload'].iloc[0],
            "speed": df['speed'].iloc[0],
            "altitude": df['altitude'].iloc[0],
            "power": integrate_flight_data(df)
        }
        flight_dict_list.append(flight_dict)
    return flight_dict_list
