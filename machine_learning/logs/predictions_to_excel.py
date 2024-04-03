import pandas as pd
import os


def predictions_to_excel(test_targets_np, test_predictions, true_power, predicted_power):
    df = pd.DataFrame({
                'True Current': test_targets_np[:, 0],
                'True Voltage': test_targets_np[:, 1],
                'True Power': true_power,
                'Predicted Voltage': test_predictions[:, 0],
                'Predicted Current': test_predictions[:, 1],
                'Predicted Power': predicted_power,
            })

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    file = os.path.join(PROJECT_ROOT, "machine_learning/logs/evaluation_results.xlsx")
    # Write data to Excel file
    with pd.ExcelWriter(file) as writer:
        df.to_excel(writer, index=False, sheet_name='Data')