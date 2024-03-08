from data_processing.energy_consumption import trapeziod_integration
class TrainingDataset:
    pass


# Takes df parameter
def target_variable_processing(dataframe):
    flight_energy = trapeziod_integration.integrate_flight_data(dataframe)



def pre_process_flights():
    pass


def organize_data():
    pass
