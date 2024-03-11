import pandas as pd
from sklearn.model_selection import train_test_split


def load_data (file_path):
    df = pd.read_csv(file_path, sep=',', low_memory=False)
    return df


def extract_flights (df):
    flights_list = [group for _, group in df.groupby('flight')]
    return flights_list


def split_data (df, test_size=0.30, val_size=0.5, random_state=42) :
    flights_list = extract_flights(df)
    # Perform the train-test split on the extracted sentences
    train_data, temp_data = train_test_split(flights_list, test_size=test_size, random_state=random_state)

    # Check if there are enough samples left for validation
    if len(temp_data) < 2:
        raise ValueError("Not enough samples remaining for validation. Adjust test_size or add more data.")

    val_data, test_data = train_test_split(temp_data, test_size=val_size, random_state=random_state)

    return train_data, val_data, test_data


def pre_process_and_split_data(file_path):
    df = load_data(file_path)
    train_data, val_data, test_data = split_data(df)
    return train_data, val_data, test_data

