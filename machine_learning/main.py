import os
from machine_learning.pre_processing import pre_process_and_split_data
from machine_learning.prepare_for_training import organize_data
from machine_learning.training import train_data


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
flights_processed = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")


def train():
    input_file = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")
    train_data, val_data, test_data = pre_process_and_split_data(input_file)

    #organizing
    train_data = organize_data(train_data)
    val_data = organize_data(val_data)
    test_data = organize_data(test_data)

    #training
    train_data(train_data)

if __name__ == "__main__":
    train()