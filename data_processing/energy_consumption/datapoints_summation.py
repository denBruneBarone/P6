import numpy as np


def add_power_to_df(df):
    df['time_interval'] = df['time'].diff().fillna(0)
    # Unit of power
    df['power_w'] = df['battery_voltage'] * df['battery_current']
    # Unit of energy
    df['energy_j'] = df['battery_voltage'] * df['battery_current'] * df['time_interval']
    df['cumulative_power'] = np.cumsum(df['energy_j'])
    df.drop(columns=['time_interval'], inplace=True)
    return df
