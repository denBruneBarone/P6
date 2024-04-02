import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from machine_learning.prepare_for_training import TrainingDataset
from machine_learning.config import HPConfig, GridSearchConfig
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from machine_learning.grid_search_logs.log import log_score
import pandas as pd


def rmse(true, predicted):  # order of params important!
    return np.sqrt(mean_squared_error(true, predicted))


def mae(true, predicted):
    return mean_absolute_error(true, predicted)


def power(true_labels, predicted_labels):
    true_power = true_labels[:, 0] * true_labels[:, 1]
    predicted_power = predicted_labels[:, 0] * predicted_labels[:, 1]

    return true_power, predicted_power


def custom_scoring_rmse(y_true, y_pred):
    rmse_current = rmse(y_true[:, 0], y_pred[:, 0])
    rmse_voltage = rmse(y_true[:, 1], y_pred[:, 1])
    return (rmse_current + rmse_voltage) / 2


def custom_scoring_mae(y_true, y_pred):
    mae_current = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_voltage = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    return (mae_current + mae_voltage) / 2


# greater_is_better=False sign-swaps the result!
custom_scoring = make_scorer(custom_scoring_mae, greater_is_better=False)


def extract_features_and_targets(dataset):
    features_list = []
    targets_list = []
    for index in range(len(dataset)):
        features, targets = dataset[index]
        features_list.append(features)
        targets_list.append(targets)

    return features_list, targets_list


def concat_1st_axis(list1, list2):
    return np.concatenate(list1, axis=0), np.concatenate(list2, axis=0)


def train_model(train_data, test_data, use_grid_search):
    grid_search_results = None
    print("Training...")
    if use_grid_search:
        print("Starting Grid Search...")
        training_dataset = TrainingDataset(train_data)

        train_features, train_targets = extract_features_and_targets(training_dataset)
        train_features_np, train_targets_np = concat_1st_axis(train_features, train_targets)

        model = DecisionTreeRegressor()
        cv = KFold(n_splits=5, shuffle=True, random_state=42)  # TODO best n_split?
        grid_search = GridSearchCV(estimator=model, param_grid=GridSearchConfig.param_grid,
                                   cv=cv, scoring=custom_scoring, verbose=2)

        grid_search.fit(train_features_np, train_targets_np)
        best_params = grid_search.best_params_
        best_score = abs(grid_search.best_score_) #abs because of GreaterIsBetter

        print("Best Params: ", best_params)
        print("Best score: ", best_score)
        grid_search_results = {"score": best_score, "params": dict(best_params)}

        model = grid_search.best_estimator_

        model.fit(train_features_np, train_targets_np)
        # TODO: Tilføj print detaljer?

    else:
        print("Training without GridSearch...")
        training_dataset = TrainingDataset(train_data)

        # Instantiate the decision tree model with specified hyperparameters
        model = DecisionTreeRegressor(criterion=HPConfig.criterion, max_depth=HPConfig.max_depth,
                                      max_features=HPConfig.max_features, max_leaf_nodes=HPConfig.max_leaf_nodes,
                                      random_state=42)

        train_features, train_targets = extract_features_and_targets(training_dataset)
        train_features_np, train_targets_np = concat_1st_axis(train_features, train_targets)

        model.fit(train_features_np, train_targets_np)

    return evaluate_model(model, test_data, grid_search_results)


def evaluate_model(model, test_data, grid_search_results=None):
    filename = 'evaluation_results.xlsx'
    print("Evaluating...")
    test_dataset = TrainingDataset(test_data)

    test_features, test_targets = extract_features_and_targets(test_dataset)
    test_features_np, test_targets_np = concat_1st_axis(test_features, test_targets)

    model.fit(test_features_np, test_targets_np)

    test_predictions = model.predict(test_features_np)

    # Calculate true and predicted power
    true_power, predicted_power = power(test_targets_np, test_predictions)
    print("True Power:", true_power)
    print("Predicted", predicted_power)

    mae = mean_absolute_error(test_targets_np, test_predictions)
    mae_power = mean_absolute_error(true_power, predicted_power)
    print("Mean Absolute Error:", mae)
    print("Mean Absolute Error Power :", mae_power)


    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'True Current': test_targets_np[:, 0],
        'True Voltage': test_targets_np[:, 1],
        'True Power': true_power,
        'Predicted Voltage': test_predictions[:, 0],
        'Predicted Current': test_predictions[:, 1],
        'Predicted Power': predicted_power,
    })

    # Write data to Excel file
    with pd.ExcelWriter(filename) as writer:
        df.to_excel(writer, index=False, sheet_name='Data')

    print(f"Evaluation results saved to {filename}")

    return model