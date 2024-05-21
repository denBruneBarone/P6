import pandas as pd
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
flights_processed = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")


def read_csv():
    data = pd.read_csv(flights_processed, sep=',')
    return data


def calculate_energy_by_summing_wh(data):
    # Calculate power consumption for each row
    data['power'] = data['battery_voltage'] * data['battery_current']
    # Group data by flight and sum power consumption for each flight
    flight_groups = data.groupby('flight')
    flight_energy = {}
    for flight, flight_data in flight_groups:
        total_energy = 0
        total_energy_wh = 0
        prev_time = flight_data['time'].iloc[0]
        for index, row in flight_data.iterrows():
            time_interval = row['time'] - prev_time
            total_energy += row['power'] * time_interval
            prev_time = row['time']
            # In wh
            total_energy_wh = total_energy / 3600
        flight_energy[flight] = total_energy_wh
    return flight_energy


def calculate_energy_by_summing_watts(data):
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
            total_energy += row['power'] * time_interval.total_seconds()  # Convert time interval to seconds
            prev_time = row['time']

        flight_energy[flight] = total_energy

    return flight_energy


def plot_connected_graph(data_x, data_y, label, color, linestyle='-'):
    plt.plot(data_x, data_y, linestyle, color=color, marker='o', label=label)


def plot_stacked_chart(data_summing, data_integrated):
    plt.figure(figsize=(50, 30))

    # Plot connected graph for data_summing
    plt.plot(list(data_summing.keys()), list(data_summing.values()), color='blue', marker='o', label='Data Summing')

    # Plot connected graph for data_integrated
    plt.plot(list(data_integrated.keys()), list(data_integrated.values()), color='orange', marker='o',
             label='Data Integrated')

    # Stacked chart
    for flight in data_summing.keys():
        plt.fill_between([flight], data_summing[flight], data_integrated[flight], color='gray', alpha=0.5)

    plt.xlabel('Flight')
    plt.ylabel('Energy Consumption (Wh)')
    plt.title('Total Energy Consumption for Each Flight')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()


data_summing_joules = calculate_energy_by_summing_wh(read_csv())

print(f"CALCULATION ENERGY CONSUMPTION FOR ALL FLIGHTS")
print("")

# Summation
print("-----------------------------------------------")
print("Calculate using summation")
print("-----------------------------------------------")
for key, value in data_summing_joules.items():
    print(f"Flight {key}: ", '%.2f' % value, 'WH', sep='')  # Using string concatenation
print("-----------------------------------------------")
print(f"Total number of flights {data_summing_joules.__len__()}")
print("")

# plot_stacked_chart(data_summing, data_integrated)
