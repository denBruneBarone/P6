import pandas as pd


def read_single_dataset():
    # Read data from CSV
    url = '../data/datasets/rodrigues/flights/1.csv'
    data = pd.read_csv(url, sep=',')

    # Calculate power consumption for each row
    data['power'] = data['battery_voltage'] * data['battery_current']

    # Integrate power over time
    total_energy = 0
    prev_time = data['time'].iloc[0]
    for index, row in data.iterrows():
        time_interval = row['time'] - prev_time
        total_energy += row['power'] * time_interval
        prev_time = row['time']

    print("Total energy consumption:", total_energy, "Joules")
    print("Total energy consumption:", (total_energy / 3600), "Watthours")


def read_multiple_dataset():
    # Read data from CSV
    url = '../data/datasets/rodrigues/flights.csv'
    data = pd.read_csv(url, sep=',')

    # Calculate power consumption for each row
    data['power'] = data['battery_voltage'] * data['battery_current']

    # Group data by flight and sum power consumption for each flight
    flight_groups = data.groupby('flight')
    flight_energy = {}
    for flight, flight_data in flight_groups:
        total_energy = 0
        prev_time = flight_data['time'].iloc[0]
        for index, row in flight_data.iterrows():
            time_interval = row['time'] - prev_time
            total_energy += row['power'] * time_interval
            prev_time = row['time']
        flight_energy[flight] = total_energy

    # Print total energy consumption for each flight
    for flight, energy in flight_energy.items():
        # print(f"Flight {flight} - Total energy consumption: {energy} Joules")
        print(f"Flight {flight} - Total energy consumption: {energy / 3600:.2f} Watthours")


# read_single_dataset()
read_multiple_dataset()
