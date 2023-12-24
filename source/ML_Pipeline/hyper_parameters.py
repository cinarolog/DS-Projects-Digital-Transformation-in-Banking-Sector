def choose_param_grid(param_grid):
    if param_grid == "logistic_param_grid":
        # logistic regression
        logistic_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                               'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        return logistic_param_grid

    elif param_grid == "naive_bayes_param_grid":
        # naive bayes
        naive_bayes_param_grid = {}  # Naive Bayes modelleri genellikle optimize edilecek belirli bir parametreye sahip değildir.
        return naive_bayes_param_grid

    elif param_grid == "svm_param_grid":
        # support vector machine
        svm_param_grid = {'C': [0.1, 1, 10, 100],
                          'gamma': [1, 0.1, 0.25, 0.01],
                          'kernel': ['rbf', 'poly', 'sigmoid']}
        return svm_param_grid

    elif param_grid == "dt_param_grid":
        # decision tree
        dt_param_grid = {'criterion': ['gini', 'entropy'],
                         'splitter': ['best', 'random'],
                         'max_depth': [None, 10, 20, 30, 40, 50],
                         'min_samples_split': [2, 5, 10],
                         'min_samples_leaf': [1, 2, 4]}
        return dt_param_grid

    elif param_grid == "rfcl_param_grid":
        # random forest
        rfcl_param_grid = {'n_estimators': [10, 50, 100, 200],
                           'criterion': ['gini', 'entropy'],
                           'max_depth': [None, 10, 20, 30, 40, 50],
                           'min_samples_split': [2, 5, 10],
                           'min_samples_leaf': [1, 2, 4]}
        return rfcl_param_grid

    elif param_grid == "xgboost_param_grid":
        # xgboost
        xgboost_param_grid = {'n_estimators': [50, 100, 200],
                              'learning_rate': [0.01, 0.1, 0.2],
                              'max_depth': [3, 5, 7],
                              'subsample': [0.8, 1.0],
                              'colsample_bytree': [0.8, 1.0]}
        return xgboost_param_grid

    elif param_grid == "knn_param_grid":
        # k-nearest neighbors
        knn_param_grid = {'n_neighbors': [3, 5, 7, 9],
                          'weights': ['uniform', 'distance'],
                          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
        return knn_param_grid

    elif param_grid == "adaboost_param_grid":
        # adaboost
        adaboost_param_grid = {'n_estimators': [50, 100, 200],
                               'learning_rate': [0.01, 0.1, 0.2, 0.5]}
        return adaboost_param_grid

    else:
        raise ValueError("Invalid param_grid value. Please provide a valid parameter grid.")
