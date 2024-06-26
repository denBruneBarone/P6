class HPConfig:
    criterion = 'friedman_mse'
    max_depth = 7  # paper siger 7
    max_features = None
    max_leaf_nodes = 10


class GridSearchConfig:
    param_grid = {
        'criterion': ['friedman_mse', 'squared_error'],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],
        'max_features': [None, 'sqrt', 'log2'],
        'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
