import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
import joblib
import os

# local modules
from src.models.naive_bayes import GaussianNaiveBayes
from src.models.decision_tree import DecisionTreeClassifier
from src.models.mlp_main import train_mlp, predict_mlp, load_base_mlp
from src.models.mlp_depth_experiment import train_and_eval_depth

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
    plt.yticks(ticks=np.arange(10), labels=CIFAR10_LABELS, fontsize=10)

    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.subplots_adjust(bottom=0.25)

    filename = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    plt.savefig(f"confusion_matrices/{filename}.png")
    plt.close()



"""
Main function that loads features.
Trains all models and saves them to disk.
If the program is run again, skips the training and uses
the saved model. Tests model variations.
Reports accuracy, precision, recall and f1.
"""
def main():

    # load features
    print("\n=== Loading Features ===")
    X_train, y_train, X_test, y_test = load_features()



    # gaussian naive bayes model that we coded
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



    # gaussian model from sci kit learn
    print("\n=== Training Sci-Kit Learn Gaussian Naive Bayes ===")
    sklearn_gnb_path = "saved_models/sklearn_gnb.pkl"

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



    # decision tree that we coded
    print("\n=== Training Custom Decision Tree ===")
    dt_path = "saved_models/custom_decision_tree.pkl"

    if os.path.exists(dt_path):
        print("Loading saved Decision Tree model...")
        dt = joblib.load(dt_path)
    else:
        print("Training Decision Tree (max_depth=50) and saving...\n (Results may take a few minutes.)")
        dt = DecisionTreeClassifier(max_depth=50)
        dt.fit(X_train, y_train)
        joblib.dump(dt, dt_path)

    y_pred_dt = dt.predict(X_test)
    full_report("Custom Decision Tree", y_test, y_pred_dt)



    # decision tree depth 
    print("\n=== Decision Tree Depth Experiment ===")
    depths = [2, 5, 10, 20, 30]

    X_small = X_train[:2000]
    y_small = y_train[:2000]

    depth_results = []

    for d in depths:
        print(f"\nTraining Decision Tree with max_depth = {d}")
        dt = DecisionTreeClassifier(max_depth=d)
        dt.fit(X_small, y_small)
        y_pred_d = dt.predict(X_test)

        acc_d = (y_pred_d == y_test).mean()
        depth_results.append((d, acc_d))
        print(f"Test accuracy at depth {d}: {acc_d:.3f}")

    print("\nSummary of depth experiment:")
    for d, acc in depth_results:
        print(f"Depth {d} → accuracy = {acc:.3f}")

    # sci kit learn tree
    from src.models.sklearn_decision_tree import run_sklearn_decision_tree

    print("\n=== Training Scikit-Learn Decision Tree (max_depth=50) ===")
    y_true_dt, y_pred_dt = run_sklearn_decision_tree()
    full_report("Scikit-Learn Decision Tree (Depth 50)", y_true_dt, y_pred_dt)

    print("\n=== Training Base MLP ===")

    base_mlp, y_pred_base = load_base_mlp(X_train, y_train, X_test, y_test)
    full_report("MLP Base Architecture", y_test, y_pred_base)

    print("\n=== MLP Depth Experiment ===")

    depth_values = [1, 2, 3, 4]
    depth_summary = []

    for d in depth_values:
        print(f"\nTraining MLP with depth = {d} hidden layer(s)")
        acc_d, preds_d = train_and_eval_depth(d, X_train, y_train, X_test, y_test)
        full_report(f"MLP Depth {d}", y_test, preds_d)
        depth_summary.append((d, acc_d))

    print("\nSummary of MLP Depth Experiment:")
    for d, acc in depth_summary:
        print(f"Depth {d}: accuracy = {acc:.3f}")

    print("\n=== MLP Hidden Size experiment ===")

    from src.models.mlp_width_experiment import load_or_train_width, predict_width

    hidden_sizes = [128, 256, 512, 1024]
    width_summary = []

    for H in hidden_sizes:
        print(f"\nTraining MLP with hidden size H = {H}")

        model_H = load_or_train_width(H, X_train, y_train)
        preds_H = predict_width(model_H, X_test)

        acc_H = (preds_H == y_test).mean()
        width_summary.append((H, acc_H))

        full_report(f"MLP Hidden Size {H}", y_test, preds_H)

    print("\nSummary of MLP Hidden Size Experiment:")
    for H, acc in width_summary:
        print(f"H={H}: accuracy = {acc:.3f}")



    print("\n=== DONE. Confusion matrices saved as PNG files. ===")



if __name__ == "__main__":
    main()
