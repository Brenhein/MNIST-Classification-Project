"""
This file is responsible for classifying a point based on the parzen window.
It uses a gaussian window to estimate the density of a point based on its
neighbors

Author: Brenden Hein
"""
from operator import itemgetter
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy.stats as sp


def hn(train, h, x, d):
    """For a given set of points (from a normal distribution), the resulting
    estimatiated denisity for a given x using a gaussian kernel function
    train (list): the training patterns
    h (int): the window size
    x (list): a test point
    d: the dimensionality
    returns (list): the window functions results based on the training data
    """
    
    dist, densities = sp.multivariate_normal(np.zeros(d), np.identity(d)), []
    for w, patterns in enumerate(train): # estimate the density of x using generated points
        px = 0
        for v in patterns:
            px += dist.pdf((np.subtract(x,v))/h)
        densities.append((px / (len(train)*(h**d)), w))
    return densities


def error(predicted, actual):
    """Gets the confusion matrix and the error rate for the test set
    predicted (list): the predicted classes
    actual (list): the actual class labels
    returns (float): the error rate
    """
    cm = confusion_matrix(actual, predicted)
    w = sum([cm[i][j] for i in range(len(cm)) for j in range(len(cm[i])) if i != j])
    print(cm, "\n")
    print("Error:", w/len(actual), "\n\n")
    return w/len(actual)


def parzen(train, test, h, d):
    """Given a window size, this function uses a gaussian parzen window to
    estimate the density of a testing point, given a cluster of training points
    train (list): the training patterns for each class
    test (list): the testing points
    h (int): the window size
    d: the dimensionality
    returns (float): the error rate
    """
    predicted, actual = [], []
    for w, patterns in enumerate(test):
        for pat in patterns:
            d_est = hn(train, h, pat, d)
            best = max(d_est, key=itemgetter(0))[1]
            predicted.append(best)
            actual.append(w) 
    return error(predicted, actual)