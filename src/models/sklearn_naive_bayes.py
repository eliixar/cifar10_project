import numpy as np
from sklearn.naive_bayes import GaussianNB

"""
Run the Sci-Kit Learn Gaussian Naive Bayes model.
Train it using our training data set. Compare its
predictions on the test set to the actual test set values.
Return the accuracy (mean of comparisons) and the predictions.
"""
def run_sklearn_gnb():
    # load PCA-reduced features
    X_train = np.load("features/X_train_50.npy") # sample sets have 50 features per image
    y_train = np.load("features/y_train.npy") # class sets were unchanged
    X_test = np.load("features/X_test_50.npy")
    y_test = np.load("features/y_test.npy")

    # create the model
    model = GaussianNB()

    # fit model to training data
    model.fit(X_train, y_train)

    # predict on test set
    y_pred = model.predict(X_test)

    # compare model prediction to actual test set
    accuracy = (y_pred == y_test).mean() # take the mean of all predictions
    print("Scikit-Learn GaussianNB Accuracy:", accuracy)

    return accuracy, y_pred

if __name__ == "__main__":
    run_sklearn_gnb()