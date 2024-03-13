import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    df = pd.read_csv(file_path, sep=',', low_memory=False)
    return df


def extract_flights(df):
    flights_list = [group for _, group in df.groupby('flight')]
    return flights_list


def split_data(df, train_size=0.8, random_state=42):
    flights_list = extract_flights(df)
    train_data, test_data = train_test_split(flights_list, test_size=1-train_size, random_state=random_state)
    return train_data, test_data


def pre_process_and_split_data(file_path):
    df = load_data(file_path)
    train_data, test_data = split_data(df)
    return train_data, test_data

