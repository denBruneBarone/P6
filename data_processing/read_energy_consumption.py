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


def calculate_average_power(data):
    # Calculate power consumption for each row
    data['power'] = data['battery_voltage'] * data['battery_current']

    # Group data by flight
    flight_groups = data.groupby('flight')
    flight_average_power = {}

    # For each flight grouped in the dataset
    for flight, flight_data in flight_groups:
        # Calculate the mean power for the flight
        average_power = flight_data['power'].mean()
        flight_average_power[flight] = average_power

    return flight_average_power


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


data = calculate_average_power(read_csv())
unit = 'W'

print(f"CALCULATING AVERAGE ENERGY CONSUMPTION FOR ALL FLIGHTS")
print("")

# Summation
print("-----------------------------------------------")
print("Calculate using summation")
print("-----------------------------------------------")
for key, value in data.items():
    print(f"Flight {key}: ", '%.2f' % value, unit, sep='')  # Using string concatenation
print("-----------------------------------------------")
print(f"Total number of flights {len(data)}")
print("")

# plot_stacked_chart(data_summing, data_integrated)

print(f"CALCULATING AVERAGE ENERGY CONSUMPTION ACROSS ALL FLIGHTS")
print("-----------------------------------------------")
print("Calculate using summation")
print("-----------------------------------------------")
sum_value = sum(data.values())
len_value = len(data)
average = sum_value / len_value
print(f"Average: ", '%.2f' % average, unit, sep='')  # Using string concatenation
print("-----------------------------------------------")
print(f"Total number of flights {len(data)}")
print("")