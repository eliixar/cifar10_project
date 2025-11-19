import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB

# local modules
from src.models.naive_bayes import GaussianNaiveBayes
from src.evaluate import evaluate_model
from src.pca_reduce import apply_pca

"""
Load the reduced PCA features of size 50x1 and return them.
"""
def load_features():
    X_train = np.load("features/X_train_50.npy")
    y_train = np.load("features/y_train.npy")
    X_test  = np.load("features/X_test_50.npy")
    y_test  = np.load("features/y_test.npy")
    return X_train, y_train, X_test, y_test

"""
Prints a report for the model beind tested, including all needed metrics.
Plots the confusion matrix and saves it to a png file.
"""
def full_report(name, y_true, y_pred):
    print("\n==============================")
    print(f"RESULTS: {name}")
    print("==============================")
    print("Accuracy :", (y_true == y_pred).mean())
    print("Precision:", precision_score(y_true, y_pred, average="macro"))
    print("Recall   :", recall_score(y_true, y_pred, average="macro"))
    print("F1 Score :", f1_score(y_true, y_pred, average="macro"))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix – {name}")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    filename = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    plt.savefig(f"confusion_matrices/{filename}.png") # save cm to a folder
    plt.close()


# main

def main():

    print("\n=== Loading Features ===")
    X_train, y_train, X_test, y_test = load_features()


    print("\n=== Training Custom Gaussian Naive Bayes ===")
    custom_gnb = GaussianNaiveBayes()
    custom_gnb.fit(X_train, y_train)
    y_pred_custom = custom_gnb.predict(X_test)

    full_report("Custom GaussianNB", y_test, y_pred_custom)

    print("\n=== Training Sci-Kit Learn Gaussian Naive Bayes ===")
    sklearn_gnb = GaussianNB()
    sklearn_gnb.fit(X_train, y_train)
    y_pred_sklearn = sklearn_gnb.predict(X_test)

    full_report("Scikit-Learn GaussianNB", y_test, y_pred_sklearn)

    print("\n=== DONE. Confusion matrices saved as PNG files. ===")


if __name__ == "__main__":
    main()
