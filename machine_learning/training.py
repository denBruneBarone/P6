import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from machine_learning.prepare_for_training import TrainingDataset
from sklearn.metrics import mean_squared_error
from machine_learning.config import HPConfig, GridSearchConfig
from sklearn.metrics import make_scorer


def rmse_cum_power(true_labels, predicted_labels):
    time_diff = np.diff(true_labels[:, 0], prepend=0)
    true_cumulative_power = np.cumsum(predicted_labels[:, 0] * predicted_labels[:, 1] * time_diff)
    test_targets_cumulative_power = np.cumsum(true_labels[:, 0] * true_labels[:, 1] * time_diff)

    rmse = np.sqrt(mean_squared_error(test_targets_cumulative_power, true_cumulative_power))
    return rmse


def custom_scoring(true_labels, predicted_labels):
    return rmse_cum_power(true_labels, predicted_labels)

custom_scoring = make_scorer(custom_scoring)


def train_model(train_data, grid_search_cv=True):
    if grid_search_cv:
        training_dataset = TrainingDataset(train_data)

        # Instantiate the decision tree model with specified hyperparameters
        model = DecisionTreeRegressor()
        # splits the train-test data into n_splits number of subsets for cross validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)  # TODO best n_split?
        grid_search = GridSearchCV(estimator=model, param_grid=GridSearchConfig.param_grid,
                                   cv=cv, scoring=custom_scoring, verbose=2)

        train_features = []
        train_targets = []

        for index in range(len(training_dataset)):
            input_array, target_array = training_dataset[index]
            train_features.append(input_array)
            train_targets.append(target_array)

        # Concatenate the lists along the appropriate axis
        train_features_np = np.concatenate(train_features, axis=0)
        train_targets_np = np.concatenate(train_targets, axis=0)

        # Fit the decision tree model
        grid_search.fit(train_features_np, train_targets_np)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print("Best Params: ", best_params)
        print("Best score: ", best_score)

        best_dt_model = grid_search.best_estimator_
        best_dt_model.fit(train_features_np, train_targets_np)
        # TODO: Tilføj print detaljer

    else:
        training_dataset = TrainingDataset(train_data)

        # Instantiate the decision tree model with specified hyperparameters
        model = DecisionTreeRegressor(criterion=HPConfig.criterion, max_depth=HPConfig.max_depth,
                                      max_features=HPConfig.max_features, max_leaf_nodes=HPConfig.max_leaf_nodes,
                                      random_state=42)

        # Extract features and targets from the training dataset
        train_features = []
        train_targets = []
        for index in range(len(training_dataset)):
            input_array, target_array = training_dataset[index]
            train_features.append(input_array)
            train_targets.append(target_array)

        # Concatenate the lists along the appropriate axis
        train_features_np = np.concatenate(train_features, axis=0)
        train_targets_np = np.concatenate(train_targets, axis=0)

        # Fit the decision tree model
        model.fit(train_features_np, train_targets_np)

    return model


def evaluate_model(model, test_data):
    test_dataset = TrainingDataset(test_data)

    print("Evaluating...")
    # Extract features and targets from the test dataset
    test_features = []
    test_targets = []
    for index in range(len(test_dataset)):
        test_input_array, test_target_array = test_dataset[index]
        test_features.append(test_input_array)
        test_targets.append(test_target_array)

    # Concatenate the lists along the appropriate axis
    test_features_np = np.concatenate(test_features, axis=0)
    test_targets_np = np.concatenate(test_targets, axis=0)

    model.fit(test_features_np, test_targets_np)

    # Predict on the test set
    test_predictions = model.predict(test_features_np)

    # Calculate RMSE for the two output parameters
    test_rmse = np.sqrt(mean_squared_error(test_targets_np, test_predictions))
    print(f"Test Root Mean Squared Error (RMSE) for Voltage and Current: {test_rmse}")

    # Calculate power consumption predictions by multiplying voltage and current predictions
    power_consumption_predictions = test_predictions[:, 0] * test_predictions[:, 1]

    # Calculate the time difference between each timestamp
    time_diff = np.diff(test_features_np[:, 0], prepend=0)

    # Adjust power consumption predictions by multiplying with the time difference
    adjusted_power_consumption = power_consumption_predictions * time_diff

    # Calculate cumulative power consumption for each flight
    cumulative_power = np.cumsum(adjusted_power_consumption)

    # Calculate RMSE on the original cumulative power for the test set
    test_targets_cumulative_power = np.cumsum(test_targets_np[:, 0] * test_targets_np[:, 1] * time_diff)
    original_test_rmse = np.sqrt(mean_squared_error(test_targets_cumulative_power, cumulative_power))

    print(f"Original Test Root Mean Squared Error (RMSE) for Cumulative Power: {original_test_rmse}")

    # Calculate RMSE on the adjusted cumulative power for the test set
    test_rmse = np.sqrt(
        mean_squared_error(test_targets_np[:, 0] * test_targets_np[:, 1], power_consumption_predictions))
    print(f"Adjusted Test Root Mean Squared Error (RMSE) for Cumulative Power: {test_rmse}")

    print("Evaluation finished!")

    return model
