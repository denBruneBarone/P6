class HPConfig:
    criterion = 'friedman_mse'  # samme som paper
    max_depth = 10  # værdien fra paperet om modeller er 7
    max_features = None  # samme som paper
    max_leaf_nodes = 500  # værdien fra paperet om modeller er 10

class BestHPConfig:
    criterion = None
    max_depth = None
    max_features = None
    max_leaf_nodes = None

class GridSearchConfig:
    param_grid = {
        'criterion': ['friedman_mse', 'squared_error'],
        'max_depth': [2, 3, 4, 5, 6, 7, 8],
        'max_features': [None, 'sqrt', 'log2'],
        'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10]
    }

