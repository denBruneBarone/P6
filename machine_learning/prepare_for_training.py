import torch
from torch.utils.data import Dataset
from data_processing.energy_consumption import trapeziod_integration

class TrainingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index].values
        sample = torch.tensor(sample, dtype=torch.float)
        return sample
    pass


# Takes df parameter
def target_variable_processing(dataframe):
    flight_energy = trapeziod_integration.integrate_flight_data(dataframe)



def organize_data(array_of_df):

    flight_dict_list = []

    for df in array_of_df:
        df['position_x'] = df['position_x'] - df['position_x'].first()
        df['position_y'] = df['position_y'] - df['position_y'].first()
        df['position_z'] = df['position_z'] - df['position_z'].first()

        flight_dict = {
            "flight": df['flight'].first(),
            "data": df,
            "payload": df['payload'].first(),
            "speed": df['speed'].first(),
            "altitude": df['altitude'].first()
        }
        flight_dict_list.append(flight_dict)
    return flight_dict_list
