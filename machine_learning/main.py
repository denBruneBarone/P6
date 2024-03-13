import os
from machine_learning.pre_processing import pre_process_and_split_data
from machine_learning.prepare_for_training import format_data
from machine_learning.training import training_and_evaluating


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
flights_processed = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")


def train():
    #pre_processing
    print("Pre-processing data...")
    input_file = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")


    #organizing
    print("Splitting data...")
    train_data, test_data = pre_process_and_split_data(input_file)
    print("Formatting data...")
    train_data = format_data(train_data)
    test_data = format_data(test_data)

    #training
    print("Training...")
    training_and_evaluating(train_data, test_data)

if __name__ == "__main__":
    train()