import numpy as np
from scipy.interpolate import interp1d


def calculate_power(data):
    """Calculate power from voltage and current."""
    voltage_values = data['battery_voltage'].values
    current_values = data['battery_current'].values
    return voltage_values * current_values


def interpolate_power(time_points, power_values):
    """Interpolate missing points in the power data."""
    interpolator = interp1d(time_points, power_values, kind='linear', fill_value="extrapolate")
    new_time_points = np.linspace(time_points.min(), time_points.max(), num=len(time_points))
    return interpolator(new_time_points), new_time_points


def trapezoidal_integration(time_points, power_values):
    """Perform trapezoidal integration."""
    total_energy = 0
    n = len(time_points)
    for i in range(n - 1):
        delta_t = time_points[i + 1] - time_points[i]
        area = (power_values[i] + power_values[i + 1]) * delta_t / 2
        total_energy += area
    return total_energy


def integrate_flight_data(df):
    """Integrate power consumption for each flight and convert to watt-hours."""
    time_points = df['time'].values
    power_values = calculate_power(df)
    interpolated_power, new_time_points = interpolate_power(time_points, power_values)
    total_energy = trapezoidal_integration(new_time_points, interpolated_power)
    # total_energy_wh = total_energy / 3600  # Convert joules to watt-hours
    flight_energy = total_energy
    return flight_energy

def integrate_flight_data2(df):
    """Integrate power consumption for each flight and add interpolated power as a new column."""
    time_points = df['time'].values
    power_values = calculate_power(df)
    interpolated_power, new_time_points = interpolate_power(time_points, power_values)

    # Create a new column with interpolated power values
    df['interpolated_power'] = np.interp(df['time'], new_time_points, interpolated_power)

    # Calculate total energy using trapezoidal integration
    total_energy = trapezoidal_integration(new_time_points, interpolated_power)

    # Uncomment the following line if you want to keep the total energy as a separate variable
    # total_energy_wh = total_energy / 3600  # Convert joules to watt-hours

    return df



def integrate_specific_flight_data(data):
    """Integrate power consumption for flight 1 and convert to watt-hours."""
    flight_energy = {}
    flight_data = data[data['flight'] == 1]  # Filter data for flight 1
    time_points = flight_data['time'].values
    power_values = calculate_power(flight_data)
    interpolated_power, new_time_points = interpolate_power(time_points, power_values)
    total_energy = trapezoidal_integration(new_time_points, interpolated_power)
    # total_energy_wh = total_energy / 3600  # Convert joules to watt-hours
    flight_energy[1] = total_energy
    return flight_energy


def integrate_each_row_specific_flight_data(data):
    """Integrate power consumption for flight 1 between each consecutive pair of rows and add integrated power to DataFrame."""
    flight_data = data[data['flight'] == 1]  # Filter data for flight 1
    time_points = flight_data['time'].values
    power_values = calculate_power(flight_data)
    interpolated_power, new_time_points = interpolate_power(time_points, power_values)

    # Initialize list to store integrated power values
    integrated_power = []

    # Integrate between each consecutive pair of rows
    for i in range(len(interpolated_power) - 1):
        time_diff = new_time_points[i + 1] - new_time_points[i]
        energy_interval = (interpolated_power[i] + interpolated_power[i + 1]) / 2.0 * time_diff
        integrated_power.append(energy_interval)

    # Calculate integrated value for the last row separately
    time_diff_last_row = new_time_points[-1] - new_time_points[-2]
    energy_interval_last_row = interpolated_power[-1] * time_diff_last_row
    integrated_power.append(energy_interval_last_row)

    # Ensure that the length of integrated power matches the length of flight data
    integrated_power.extend([None] * (len(flight_data) - len(integrated_power)))

    # Add integrated power as a new column to the DataFrame
    flight_data['integrated_power'] = integrated_power

    return flight_data


def add_cumulative_column(data):
    """Add a cumulative sum column to the flight data."""
    data['cumulative'] = data['integrated_power'].cumsum()
    return data

