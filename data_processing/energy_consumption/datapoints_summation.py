def calculate_energy_by_summing(data):

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
            total_energy_wh = total_energy / 3600
        flight_energy[flight] = total_energy_wh
    return flight_energy



