class HPConfig:
    criterion = 'friedman_mse'
    max_depth = 6 # paper siger 7
    max_features = None
    max_leaf_nodes = 10


class BestHPConfig:
    criterion = 'friedman_mse'
    max_depth = 6
    max_features = None
    max_leaf_nodes = 10


class GridSearchConfig:
    param_grid = {
        'criterion': ['friedman_mse', 'squared_error'],
        'max_depth': [5, 6, 7, 8],
        'max_features': [None],
        'max_leaf_nodes': [8, 9, 10]
    }

