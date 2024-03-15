import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from machine_learning.config import GridSearchConfig
from machine_learning.prepare_for_training import format_data


def grid_search(array_of_df):
    # n_splits: number of subsets,
    # splits the train-val data into n_splits number of subsets for cross validation
    decisionTree = DecisionTreeRegressor()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(estimator=decisionTree, param_grid=GridSearchConfig.param_grid,
                               cv=cv, scoring='friedman_mse')

    flight_dict_list = format_data(array_of_df)

    # Extract features and target variable from flight_dict_list
    x_train_list = [flight['data'] for flight in flight_dict_list]
    y_train_list = [flight['power'] for flight in flight_dict_list]

    # Convert lists of DataFrames/Series into a single DataFrame and Series
    x_train = pd.concat(x_train_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)

    # Perform grid search
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    best_regressor = DecisionTreeRegressor(**best_params)
    best_regressor.fit(x_train, y_train)

    return best_params, best_score, best_regressor
