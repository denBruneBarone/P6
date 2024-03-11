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


def pre_process_flights():
    pass


def organize_data():
    pass
