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
    new_time_points = np.linspace(time_points.min(), time_points.max(), num=1000)
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


def integrate_multiple_flight_data(data):
    """Integrate power consumption for each flight and convert to watt-hours."""
    flight_energy = {}
    grouped_data = data.groupby('flight')
    for flight, flight_data in grouped_data:
        time_points = flight_data['time'].values
        power_values = calculate_power(flight_data)
        interpolated_power, new_time_points = interpolate_power(time_points, power_values)
        total_energy = trapezoidal_integration(new_time_points, interpolated_power)
        total_energy_wh = total_energy / 3600  # Convert joules to watt-hours
        flight_energy[flight] = total_energy_wh
    return flight_energy

def integrate_single_flight_data(flight_data):
    """Integrate power consumption for a single flight and convert to watt-hours."""
    time_points = flight_data['time'].values
    power_values = calculate_power(flight_data)
    interpolated_power, new_time_points = interpolate_power(time_points, power_values)
    total_energy = trapezoidal_integration(new_time_points, interpolated_power)
    total_energy_wh = total_energy / 3600  # Convert joules to watt-hours
    return total_energy_wh
