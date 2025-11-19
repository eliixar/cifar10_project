import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
import joblib
import os

# local modules
from src.models.naive_bayes import GaussianNaiveBayes
from src.models.decision_tree import DecisionTreeClassifier


# class labels for matrix legend
CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat",
    "deer", "dog", "frog", "horse",
    "ship", "truck"
]


os.makedirs("saved_models", exist_ok=True)
os.makedirs("confusion_matrices", exist_ok=True)


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
Prints a report for the model being tested, including all needed metrics.
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

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix – {name}", fontsize=16)
    plt.colorbar()

    plt.xticks(ticks=np.arange(10), labels=CIFAR10_LABELS, rotation=45, ha="right", fontsize=10)
    plt.yticks(ticks=np.arange(10), labels=CIFAR10_LABELS, fontsize=10) # add class labels

    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.subplots_adjust(bottom=0.25)

    filename = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    plt.savefig(f"confusion_matrices/{filename}.png")  # save cm to a folder
    plt.close()


# main
def main():

    # load features
    print("\n=== Loading Features ===")
    X_train, y_train, X_test, y_test = load_features()


    # custom gnb model. train once, save it then use it
    print("\n=== Training Custom Gaussian Naive Bayes ===")
    custom_gnb_path = "saved_models/custom_gnb.pkl"

    if os.path.exists(custom_gnb_path):
        print("Loading saved Custom GaussianNB model...")
        custom_gnb = joblib.load(custom_gnb_path)
    else:
        print("Training Custom GaussianNB and saving...")
        custom_gnb = GaussianNaiveBayes()
        custom_gnb.fit(X_train, y_train)
        joblib.dump(custom_gnb, custom_gnb_path)

    y_pred_custom = custom_gnb.predict(X_test)
    full_report("Custom GaussianNB", y_test, y_pred_custom)


   # sci-kit learn gnb. train it once then use it
    print("\n=== Training Sci-Kit Learn Gaussian Naive Bayes ===")
    sklearn_gnb_path = "saved_models/sklearn_gnb.pkl"

    # running this program a second time will skip training and use the saved model
    if os.path.exists(sklearn_gnb_path):
        print("Loading saved Scikit-Learn GaussianNB model...") 
        sklearn_gnb = joblib.load(sklearn_gnb_path)
    else:
        print("Training Scikit-Learn GaussianNB and saving...")
        sklearn_gnb = GaussianNB()
        sklearn_gnb.fit(X_train, y_train)
        joblib.dump(sklearn_gnb, sklearn_gnb_path)

    y_pred_sklearn = sklearn_gnb.predict(X_test)
    full_report("Scikit-Learn GaussianNB", y_test, y_pred_sklearn)

        
    # custom decision tree training
    print("\n=== Training Custom Decision Tree ===")

    dt_path = "saved_models/custom_decision_tree.pkl"

    if os.path.exists(dt_path):
        print("Loading saved Decision Tree model...")
        dt = joblib.load(dt_path)
    else:
        print("Training Decision Tree (max_depth=50) and saving.../n "
        "(Results may take a few minutes to process.)")
        dt = DecisionTreeClassifier(max_depth=50)
        dt.fit(X_train, y_train)
        joblib.dump(dt, dt_path)

    y_pred_dt = dt.predict(X_test)
    full_report("Custom Decision Tree", y_test, y_pred_dt)
    
    # Decision Tree depth experiment
    print("\n=== Decision Tree Depth Experiment ===")
    depths = [2, 5, 10, 20, 30]

    # even at small depths the process takes long, so we reduce the data size
    X_small = X_train[:2000]
    y_small = y_train[:2000]

    depth_results = []

    for d in depths:
        print(f"\nTraining Decision Tree with max_depth = {d} (Results may take a few minutes to process.)")
        dt = DecisionTreeClassifier(max_depth=d)
        # use small data
        dt.fit(X_small, y_small)
        y_pred_d = dt.predict(X_test)

        acc_d = (y_pred_d == y_test).mean()
        depth_results.append((d, acc_d))
        print(f"Test accuracy at depth {d}: {acc_d:.3f}")

    print("\nSummary of depth experiment (depth, accuracy):")
    for d, acc in depth_results:
        print(f"Depth {d:2d} → accuracy = {acc:.3f}")

    #  train scikit learn decision tree 
    from src.models.sklearn_decision_tree import run_sklearn_decision_tree

    print("\n=== Training Scikit-Learn Decision Tree (max_depth=50) ===")
    y_true_dt, y_pred_dt = run_sklearn_decision_tree()
    full_report("Scikit-Learn Decision Tree (Depth 50)", y_true_dt, y_pred_dt)

    print("\n=== DONE. Confusion matrices saved as PNG files. ===")


if __name__ == "__main__":
    main()
