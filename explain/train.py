import numpy as np
from catboost import CatBoostClassifier, Pool


def train_binary_classifier(X_train, y_train, X_test, y_test):
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)

    model_params = {
        'eval_metric': 'F1',
        'custom_metric': ['Accuracy', 'Precision', 'Recall'],
    }

    grid_params = {
        'iterations': np.linspace(1000, 3000, 2, dtype=int),
        'learning_rate': np.logspace(-2, 0, 2),
        'l2_leaf_reg': np.logspace(-2, 1, 2)
    }

    model = CatBoostClassifier(**model_params, logging_level='Silent')
    cv = model.grid_search(grid_params, train_pool, plot=False, verbose=False)
    params, cv_results = cv['params'], cv['cv_results']

    params = {**model_params, **params, 'auto_class_weights': 'Balanced'}

    final_model = CatBoostClassifier(**params)
    final_model.fit(train_pool, eval_set=test_pool, plot=False, silent=True)

    return final_model
