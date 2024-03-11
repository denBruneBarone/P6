import os
from machine_learning.pre_processing import pre_process_and_split_data

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
flights_processed = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")


def train():
    input_file = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")
    train_df, val_df, test_df = pre_process_and_split_data(input_file)
    print("Val_len:", len(val_df))
    print("test_len:",len(test_df))
    print("Train_len:", len(train_df))