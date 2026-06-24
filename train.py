import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def train_and_tune_models(X_train, y_train, random_state=42):
    """
    Trains and tunes Logistic Regression, Random Forest, and K-Nearest Neighbors 
    using GridSearchCV (5-fold Cross-Validation).
    """
    models = {}
    best_params = {}
    cv_results = {}
    
    # 1. Logistic Regression
    print("\n--- Tuning Logistic Regression ---")
    log_reg = LogisticRegression(solver='saga', max_iter=5000, random_state=random_state)
    lr_param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2']
    }
    lr_grid = GridSearchCV(log_reg, lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    models['Logistic Regression'] = lr_grid.best_estimator_
    best_params['Logistic Regression'] = lr_grid.best_params_
    print(f"Best Params: {lr_grid.best_params_}")
    print(f"Best CV Accuracy: {lr_grid.best_score_:.4f}")
    
    # 2. Random Forest
    print("\n--- Tuning Random Forest ---")
    rf = RandomForestClassifier(random_state=random_state)
    rf_param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 5, 8, 12],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    models['Random Forest'] = rf_grid.best_estimator_
    best_params['Random Forest'] = rf_grid.best_params_
    print(f"Best Params: {rf_grid.best_params_}")
    print(f"Best CV Accuracy: {rf_grid.best_score_:.4f}")
    
    # 3. K-Nearest Neighbors (KNN)
    print("\n--- Tuning K-Nearest Neighbors ---")
    knn = KNeighborsClassifier()
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    knn_grid = GridSearchCV(knn, knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    knn_grid.fit(X_train, y_train)
    models['K-Nearest Neighbors'] = knn_grid.best_estimator_
    best_params['K-Nearest Neighbors'] = knn_grid.best_params_
    print(f"Best Params: {knn_grid.best_params_}")
    print(f"Best CV Accuracy: {knn_grid.best_score_:.4f}")
    
    return models, best_params
