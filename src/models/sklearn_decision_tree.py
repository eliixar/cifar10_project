import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
"""
Scikit-Learn Decision Tree for CIFAR-10 (PCA-reduced features).
Uses Gini impurity and max_depth=50.
"""
def run_sklearn_decision_tree():
    # ensure directory exists
    os.makedirs("saved_models", exist_ok=True)

    # load reduced features
    X_train = np.load("features/X_train_50.npy")
    y_train = np.load("features/y_train.npy")
    X_test  = np.load("features/X_test_50.npy")
    y_test  = np.load("features/y_test.npy")

    model_path = "saved_models/sklearn_dt.pkl"

    # load or train
    if os.path.exists(model_path):
        print("Loading saved Scikit-Learn Decision Tree model...")
        dt = joblib.load(model_path)
    else:
        print("Training Scikit-Learn Decision Tree (max_depth=50, Gini)...")
        dt = DecisionTreeClassifier(
            criterion="gini",
            max_depth=50
        )
        dt.fit(X_train, y_train)
        joblib.dump(dt, model_path)

    # predict
    y_pred = dt.predict(X_test)

    return y_test, y_pred
