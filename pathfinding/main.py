from pathfinding.Mission import Mission
from pathfinding.Workspace import Workspace
from pathfinding.Node import Node
from pathfinding.find_paths import find_baseline_path, find_optimal_path
from pathfinding.Blockage import Blockage
import warnings


def setup_workspace(mission):
    # Setup workspace with bounds and dimensions and blockages.
    dimensions = 3  # Define a 3-dimensional space
    max_bounds = [400, 400, 60]  # Define maximum bounds for each dimension
    space = Workspace(dimensions, max_bounds, mission)

    # Add Blockages
    blockage_matrix_1 = Blockage(50, 50, 50, 150, 150, 0, 'obstacle')
    space.add_blockage(blockage_matrix_1)
    blockage_matrix_2 = Blockage(50, 50, 30, 0, 100, 0, 'obstacle')
    space.add_blockage(blockage_matrix_2)
    blockage_matrix_3 = Blockage(40, 80, 10, 300, 50, 0, 'obstacle')
    space.add_blockage(blockage_matrix_3)
    blockage_matrix_4 = Blockage(40, 35, 15, 260, 95, 0, 'obstacle')
    space.add_blockage(blockage_matrix_4)
    blockage_matrix_5 = Blockage(40, 95, 25, 55, 185, 0, 'obstacle')
    space.add_blockage(blockage_matrix_5)
    blockage_matrix_6 = Blockage(100, 35, 20, 175, 200, 0, 'obstacle')
    space.add_blockage(blockage_matrix_6)
    blockage_matrix_7 = Blockage(100, 35, 10, 110, 310, 0, 'obstacle')
    space.add_blockage(blockage_matrix_7)
    space.set_wind(wind_speed=10, wind_angle=90)
    return space


def find_and_show_optimal_path():
    # Define a mission
    mission = Mission(Node(0, 0, 0), Node(400, 400, 0), 500)

    workspace = setup_workspace(mission)

    flight_optimal = None
    flight_baseline = None

    flight_optimal = find_optimal_path(workspace)
    flight_baseline = find_baseline_path(workspace)
    print_stats(flight_optimal, flight_baseline)

    workspace.add_flight_path(flight_baseline)
    workspace.add_flight_path(flight_path=flight_optimal)

    workspace.plot_space(dimension='2D', dpi=800, show_wind=False)
    workspace.plot_space(dimension='3D', dpi=800)


def print_stats(flight_optimal=None, flight_baseline=None):
    if flight_optimal is not None:
        print("optimal path", flight_optimal.path)
        print(f"optimal energy {flight_optimal.energy} Joule")
    if flight_baseline is not None:
        print("baseline path", flight_baseline.path)
        print(f"baseline energy {flight_baseline.energy} Joule")

    if flight_optimal is not None and flight_baseline is not None:
        energy_diff = flight_optimal.energy / flight_baseline.energy * 100
        if energy_diff > 100:
            warnings.warn("Baseline cheaper than optimal path!")

        print(f"optimal / baseline * 100: {energy_diff}")
    elif not flight_optimal and not flight_baseline:
        warnings.warn("Flight_path: No flight paths provided found")


if __name__ == '__main__':
    find_and_show_optimal_path()
