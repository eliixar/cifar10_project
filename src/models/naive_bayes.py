import numpy as np

class GaussianNaiveBayes:
    """
    Function that finds the mean, variance and priors of each
    feature vector in all classes of a set.
    """
    def fit(self, X, y):
        self.classes = np.unique(y) # find class labels
        # initialize empty sets to store mean, variance and priors
        self.means = {}
        self.vars = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]   # samples of class c
            self.means[c] = X_c.mean(axis=0) # compute mean of class c
            self.vars[c] = X_c.var(axis=0) + 1e-6  # add small number to avoid divide-by-zero
            # compute prior (num samples in c / total num samples)
            self.priors[c] = len(X_c) / len(X)

    """
    Function that returns the Gaussian Probability Density Function
    using the values we calculated above.
    """
    def gaussian_pdf(self, x, mean, var):
        return np.exp(-0.5 * ((x - mean) ** 2) / var) / np.sqrt(2 * np.pi * var)

    """
    Predict the class label for each sample in X.
    Computes the log-posterior for each class:
        log P(c) + sum(log Gaussian(x_i | μ_ci, σ²_ci))
    Returns an array of predictions (one per sample).
    """
    def predict(self, X):
        # initialize empty list
        preds = []

        # loop through all samples
        for x in X:
            posteriors = [] # initialize posteriors

            for c in self.classes:
                # take log of prior
                prior = np.log(self.priors[c]) 
                # use log to avoid underflow/losing small data
                likelihood = np.sum(np.log(self.gaussian_pdf(x, self.means[c], self.vars[c])))
                posteriors.append(prior + likelihood) # pick class with highest probability

            # append the most likely class c for sample x to the list 
            preds.append(np.argmax(posteriors))

        return np.array(preds)
