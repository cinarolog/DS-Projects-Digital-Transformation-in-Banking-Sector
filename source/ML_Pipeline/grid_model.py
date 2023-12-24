from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

def grid_model(X_train, y_train, param_grid, model_name):

    """
        Datalarımızı veriyoruz,modelimize 
        uygun param_grid tanımlıyoruz.
    """

    model_dict = {
        'logistic_reg': LogisticRegression,
        'naive_bayes': GaussianNB,
        'svm_model': SVC,
        'decision_tree': DecisionTreeClassifier,
        'rfcl': RandomForestClassifier,
        'xgboost': XGBClassifier,
        'knn': KNeighborsClassifier,
        'adaboost': AdaBoostClassifier,
    }

    model = model_dict[model_name]()  # any model you want
    model.fit(X_train, y_train)

    grid_search = GridSearchCV(model, param_grid, refit=True, verbose=2)
    grid_search.fit(X_train, y_train)
    final_predictor = grid_search.best_estimator_
    return final_predictor
