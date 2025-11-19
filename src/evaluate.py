import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

"""
Uses scikit-learn built-in metrics to evaluate the effectiveness
of the model passed as an argument. Prints and returns
accuracy, precision, recall, f1 and the confusion matrix.
"""
def evaluate_model(y_true, y_pred, model_name="Model"):
    # calculate accuracy
    accuracy = (y_true == y_pred).mean()

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    print(f"\n=== Evaluation: {model_name} ===")
    print("Accuracy:", accuracy)
    print("Precision per class:", precision)
    print("Recall per class:", recall)
    print("F1-score per class:", f1)
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }