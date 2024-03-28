import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from machine_learning.prepare_for_training import TrainingDataset
from sklearn.metrics import mean_squared_error
from machine_learning.config import HPConfig, BestHPConfig, GridSearchConfig
from sklearn.metrics import make_scorer
from machine_learning.log import log_score


def rmse(true, predicted): #order of params important!
    return np.sqrt(mean_squared_error(true, predicted))


def power(true_labels, predicted_labels):
    true_power = predicted_labels[:, 0] * predicted_labels[:, 1]
    predicted_power = true_labels[:, 0] * true_labels[:, 1]

    return true_power, predicted_power


def cum_power(true_power, predicted_power, time_diff):
    true_cum_power = np.cumsum(true_power * time_diff)
    predicted_cum_power = np.cumsum(predicted_power * time_diff)
    return true_cum_power, predicted_cum_power


def custom_scoring(true_labels, predicted_labels):
    true_power, predicted_power = power(true_labels, predicted_labels)
    return rmse(true_power, predicted_power)


# greater_is_better=False sign-swaps the result!
custom_scoring = make_scorer(custom_scoring, greater_is_better=False)


def train_model(train_data, test_data, is_grid_search_cv):
    grid_search_results = None
    print("Training...")
    if is_grid_search_cv:
        print("Starting Grid Search...")
        training_dataset = TrainingDataset(train_data)

        train_features = []
        train_targets = []

        for index in range(len(training_dataset)):
            input_array, target_array = training_dataset[index]
            train_features.append(input_array)
            train_targets.append(target_array)

        train_features_np = np.concatenate(train_features, axis=0)
        train_targets_np = np.concatenate(train_targets, axis=0)

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

        train_features = []
        train_targets = []
        for index in range(len(training_dataset)):
            input_array, target_array = training_dataset[index]
            train_features.append(input_array)
            train_targets.append(target_array)

        train_features_np = np.concatenate(train_features, axis=0)
        train_targets_np = np.concatenate(train_targets, axis=0)

        model.fit(train_features_np, train_targets_np)

    return evaluate_model(model, test_data, grid_search_results)


def evaluate_model(model, test_data, grid_search_results=None):
    print("Evaluating...")
    test_dataset = TrainingDataset(test_data)
    test_features = []
    test_targets = []

    for index in range(len(test_dataset)):
        test_input_array, test_target_array = test_dataset[index]
        test_features.append(test_input_array)
        test_targets.append(test_target_array)

    test_features_np = np.concatenate(test_features, axis=0)
    test_targets_np = np.concatenate(test_targets, axis=0)

    model.fit(test_features_np, test_targets_np)

    test_predictions = model.predict(test_features_np)
    rmse_targets = rmse(test_targets_np, test_predictions)
    print(f"Test Root Mean Squared Error (RMSE) for Voltage and Current: {rmse_targets}")

    true_power, predicted_power = power(test_targets_np, test_predictions)
    rmse_power = rmse(true_power, predicted_power)
    print(f"Test Root Mean Squared Error (RMSE) for Power: {rmse_power}")

    time_diff = np.diff(test_features_np[:, 0], prepend=0)
    true_cum_power, predicted_cum_power = cum_power(true_power, predicted_power, time_diff)
    rmse_cum_power = rmse(true_cum_power, predicted_cum_power)
    print(f"Test Mean Squared Error (RMSE) for Cumulative Power: {rmse_cum_power}")
    print("Evaluation finished!")

    if grid_search_results is not None:
        log_score(grid_search_results['score'], rmse_targets, rmse_power, rmse_cum_power, grid_search_results['params'])

    return model
