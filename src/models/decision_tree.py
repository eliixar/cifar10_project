import numpy as np
import joblib
import os

"""
Node structure for the Decision Tree.

Each node represents either:
    • A decision node, containing:
        - feature:      index of the feature to test
        - threshold:    numerical threshold for the split
        - left:         left subtree (samples <= threshold)
        - right:        right subtree (samples > threshold)

    • A leaf node, containing:
        - value:        the predicted class label for that leaf

Internal nodes have value=None.
Leaf nodes have feature=None and threshold=None.
"""
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


"""
Decision Tree Classifier implemented using only Python and NumPy,
as required by the assignment.

This classifier:
    • Uses the Gini impurity measure for evaluating splits.
    • Performs binary threshold splits on continuous features.
    • Recursively grows the tree up to a maximum depth.
    • Stores the tree starting from a root DecisionTreeNode.
"""
class DecisionTreeClassifier:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.root = None

    """
    Compute the Gini impurity of a label vector y.

    Formula:
        Gini = 1 − Σ (p_k)^2
    where p_k is the proportion of samples in class k.

    Pure nodes (all labels identical) have Gini = 0.
    Larger Gini means more mixed labels.
    """
    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    """
    Split the dataset into two subsets based on a selected
    feature index and threshold value.

    Returns:
        X_left,  y_left   → samples where feature <= threshold
        X_right, y_right  → samples where feature > threshold
    """
    def split(self, X, y, feature, threshold):
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

    """
    Determine the best possible split across all features.

    For each feature:
        • Extract all unique values as potential thresholds.
        • Try each threshold and compute the Gini impurity
          of the resulting split.
        • Select the split that achieves the lowest impurity.

    Returns:
        (best_feature, best_threshold)

    If no valid split exists, returns (None, None).
    """
    def best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gini = float("inf")

        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for t in thresholds:
                X_l, y_l, X_r, y_r = self.split(X, y, feature, t)

                if len(y_l) == 0 or len(y_r) == 0:
                    continue

                g = (len(y_l) / len(y)) * self.gini(y_l) + \
                    (len(y_r) / len(y)) * self.gini(y_r)

                if g < best_gini:
                    best_feature = feature
                    best_threshold = t
                    best_gini = g

        return best_feature, best_threshold

    """
    Recursively construct the decision tree.

    Stopping conditions:
        • Maximum depth reached
        • Node is pure (contains only one class)
        • No valid split exists

    Otherwise:
        • Find best split
        • Recursively build left and right subtrees
        • Create and return an internal DecisionTreeNode
    """
    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(value=leaf_value)

        feature, threshold = self.best_split(X, y)

        if feature is None:
            leaf_value = np.bincount(y).argmax()
            return DecisionTreeNode(value=leaf_value)

        X_l, y_l, X_r, y_r = self.split(X, y, feature, threshold)

        left_child = self.build_tree(X_l, y_l, depth + 1)
        right_child = self.build_tree(X_r, y_r, depth + 1)

        return DecisionTreeNode(feature, threshold, left_child, right_child)

    """
    Method for training the tree.
    Stores the root node of the fully constructed tree.
    """
    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    """
    Predict the class label for a single sample by descending
    from the root until reaching a leaf node.

    At each internal node:
        • If sample[feature] <= threshold → go left
        • Otherwise → go right
    """
    def predict_single(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)

    """
    Predict class labels for an entire dataset by calling
    predict_single on each sample.

    Returns a vector of predicted labels.
    """
    def predict(self, X):
        return np.array([self.predict_single(x, self.root) for x in X])
