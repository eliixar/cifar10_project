import numpy as np
from sklearn.decomposition import PCA

"""
Function that loads the train and test data sets.
Reduces the amount of feature vectors from 512 to 50.
Saves the new files of sizes 5000x50 and 1000x50.
"""
def apply_pca():

    # load feature vectors
    X_train = np.load("features/x_train.npy")
    y_train = np.load("features/y_train.npy")
    X_test  = np.load("features/x_test.npy")
    y_test  = np.load("features/y_test.npy")
    
    # further reduce to 50 dimensions
    pca = PCA(n_components=50)

    # fit PCA on train (only for X since y represents labels)
    X_train_reduced = pca.fit_transform(X_train)

    # apply PCA to test set
    X_test_reduced = pca.transform(X_test)

    # save results
    np.save("features/X_train_50.npy", X_train_reduced)
    np.save("features/X_test_50.npy", X_test_reduced)
    np.save("features/y_train.npy", y_train)
    np.save("features/y_test.npy", y_test)

    print("PCA complete!")
    print("Train shape:", X_train_reduced.shape) # 5000 rows, reduced to 50 features each
    print("Test shape: ", X_test_reduced.shape) # 1000 rows. 512 -> 50 columns

if __name__ == "__main__":
    apply_pca()