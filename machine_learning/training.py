import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from machine_learning.prepare_for_training import TrainingDataset
from machine_learning.logs.predictions_to_excel import predictions_to_excel
from machine_learning.config import HPConfig, GridSearchConfig
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from machine_learning.logs.hp_to_csv import hp_to_csv


def rmse(true, predicted):  # order of params important!
    return np.sqrt(mean_squared_error(true, predicted))


def mae(true, predicted):
    return mean_absolute_error(true, predicted)


def mape(true, predicted):
    epsilon = 1e-10
    true_safe = np.where(true == 0, epsilon, true)
    absolute_percentage_errors = np.abs((true - predicted) / true_safe) * 100
    return np.mean(absolute_percentage_errors)


# def mape(true, predicted):
#     absolute_percentage_errors = np.abs((true - predicted) / true) * 100
#     return np.mean(absolute_percentage_errors)

def r_squared(true, predicted):
    return r2_score(true, predicted)


# def pa_nmae_mean(true, predicted):
#     mae_score = mae(true, predicted)
#     normalized_mae = mae_score / np.mean(true)
#     percentage_accuracy_score = (1 - normalized_mae) * 100
#     return percentage_accuracy_score
#

def pa_nmae_range(true, predicted):
    mae_score = mae(true, predicted)
    target_range = np.max(true) - np.min(true)
    normalized_mae = mae_score / target_range
    percentage_accuracy_score = (1 - normalized_mae) * 100
    return percentage_accuracy_score


def power(true_labels, predicted_labels):
    true_power = true_labels[:, 0] * true_labels[:, 1]
    predicted_power = predicted_labels[:, 0] * predicted_labels[:, 1]

    return true_power, predicted_power


def custom_scoring_power(y_true, y_pred):
    true_power, pred_power = power(y_true, y_pred)
    return mae(true_power, pred_power)

def custom_scoring_rmse(y_true, y_pred):
    rmse_current = mae(y_true[:, 0], y_pred[:, 0])
    rmse_voltage = mae(y_true[:, 1], y_pred[:, 1])
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
    # Converting two lists that is array of arrays. To Two list that each is one big array. Like a csv.
    return np.concatenate(list1, axis=0), np.concatenate(list2, axis=0)


def train_model(train_data, test_data, use_grid_search):
    grid_search_results = None
    print("Training...")

    # Using __getitem__ method in TrainingDataset to get features and targets
    training_dataset = TrainingDataset(train_data)

    train_features, train_targets = extract_features_and_targets(training_dataset)
    train_features_np, train_targets_np = concat_1st_axis(train_features, train_targets)

    if use_grid_search:
        print("Starting Grid Search...")

        model = DecisionTreeRegressor()
        cv = KFold(n_splits=5, shuffle=True, random_state=42)  # TODO best n_split?
        grid_search = GridSearchCV(estimator=model, param_grid=GridSearchConfig.param_grid,
                                   cv=cv, scoring=custom_scoring, verbose=2)

        grid_search.fit(train_features_np, train_targets_np)
        best_params = grid_search.best_params_
        best_score = abs(grid_search.best_score_)  # abs because of GreaterIsBetter

        print("Best Params: ", best_params)
        print("Best score: ", best_score)
        grid_search_results = {"score": best_score, "params": dict(best_params)}

        model = grid_search.best_estimator_
        # TODO: Tilføj print detaljer?

    else:
        print("Training without GridSearch...")

        # Instantiate the decision tree model with specified hyperparameters
        model = DecisionTreeRegressor(criterion=HPConfig.criterion, max_depth=HPConfig.max_depth,
                                      max_features=HPConfig.max_features, max_leaf_nodes=HPConfig.max_leaf_nodes,
                                      random_state=42)

    model.fit(train_features_np, train_targets_np)
    return evaluate_model(model, test_data, grid_search_results)


def evaluate_model(model, test_data, grid_search_results=None, save_predictions_in_excel=True):
    def print_stats(arg1, arg2, arg3, arg4):
        print(f"Test Root Mean Squared Error (RMSE) for {arg4}: {arg1}")
        print(f"Test Mean Absolute Error (MAE) for {arg4}: {arg2}")
        print(f"Test Percentage Accuracy (PA) for {arg4}: {arg3}")

    print("Evaluating...")
    test_dataset = TrainingDataset(test_data)

    test_features, test_targets = extract_features_and_targets(test_dataset)
    test_features_np, test_targets_np = concat_1st_axis(test_features, test_targets)

    model.fit(test_features_np, test_targets_np)

    test_predictions = model.predict(test_features_np)

    try:
        rmse_targets = rmse(test_targets_np, test_predictions)
        mae_targets = mae(test_targets_np, test_predictions)
        pa_targets = pa_nmae_range(test_targets_np, test_predictions)
        # r_squared_targets = r_squared(test_targets_np, test_predictions)
        # mape_targets = mape(test_targets_np, test_predictions)
        print_stats(rmse_targets, mae_targets, pa_targets, 'Voltage and Current')

        true_power, predicted_power = power(test_targets_np, test_predictions)
        rmse_power = rmse(true_power, predicted_power)
        mae_power = mean_absolute_error(true_power, predicted_power)
        pa_power = pa_nmae_range(true_power, predicted_power)
        # r_squared_power = r_squared(true_power, predicted_power)
        # mape_power = mape(true_power, predicted_power)

        print_stats(rmse_power, mae_power, pa_power, 'Power')
    except Exception as e:
        print(f"Error: An error occured while printing the statistics. {e}")

    if save_predictions_in_excel:
        predictions_to_excel(test_targets_np, test_predictions, true_power, predicted_power)

    if grid_search_results is not None:
        hp_to_csv(grid_search_results['score'], rmse_targets, mae_targets, pa_targets, rmse_power, mae_power, pa_power,
                  grid_search_results['params'])

    return model
