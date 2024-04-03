import csv
from datetime import datetime
import getpass
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
file = os.path.join(PROJECT_ROOT, "machine_learning/grid_search_logs/hp_log.csv")


def log_score(score, rmse_targets, mae_targets, rmse_power, mae_power, params):
    # Get the username of the current user
    username = getpass.getuser()

    # Read existing entries from the CSV file
    existing_entries = []
    with open(file, 'r', newline='') as f:
        reader = csv.reader(f)
        # Skip the header row
        next(reader)
        for row in reader:
            existing_entries.append(row)

    # Add the new entry
    param_values = [value if value is not None else "None" for value in params.values()]
    new_entry = [score] + [rmse_targets] + [mae_targets] + [rmse_power] + [mae_power] + param_values + [username] + [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    existing_entries.append(new_entry)

    # Sort the entries by score
    sorted_entries = sorted(existing_entries, key=lambda x: float(x[0]))

    # Write the sorted entries back to the CSV file
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['score', 'rmse_targets', 'mae_targets', 'rmse_power', 'mae_power', 'criterion', 'max_depth', 'max_features', 'max_leaf_nodes', 'username', 'date'])
        writer.writerows(sorted_entries)

