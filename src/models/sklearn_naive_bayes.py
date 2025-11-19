import numpy as np
import joblib
from sklearn.naive_bayes import GaussianNB

"""
Run the Sci-Kit Learn Gaussian Naive Bayes model.
Loads a previously saved model and evaluates it
on the PCA-reduced test set.
"""
def run_sklearn_gnb():
    # load PCA-reduced features
    X_test = np.load("features/X_test_50.npy")
    y_test = np.load("features/y_test.npy")

    # load previously saved model
    model = joblib.load("saved_models/sklearn_gnb.pkl")

    # predict on test set
    y_pred = model.predict(X_test)

    # compute accuracy manually
    accuracy = (y_pred == y_test).mean()

    print("Scikit-Learn GaussianNB Accuracy:", accuracy)

    return accuracy, y_pred


if __name__ == "__main__":
    run_sklearn_gnb()
