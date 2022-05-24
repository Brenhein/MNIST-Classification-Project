"""
This file is responsible for using KNN to classify points with an unknown
distribution.  It calculates the distance from a testing points to all
training points, finds its k nearest neighbors, and classifies the point based
highest class occurence in its k nearest neighbors

Author: Brenden Hein
"""
from operator import itemgetter
import scipy.spatial.distance as sp
from sklearn.metrics import confusion_matrix


def classify_knn(k, distances):
    """Using K-NN, classifies the testing points USING the training points
    k (int): the humber of nearest neighbors
    distances (list): the distances from the test points to all training points
    returns (int): the predicted class label
    """
    if k == 1:
        distances_k = [min(distances, key=itemgetter(0))]
    else:
        distances_k = sorted(distances, key=itemgetter(0))[:k]
    counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for d in distances_k:
        counts[d[1]] += 1
    return max(enumerate(counts), key=itemgetter(1))[0]+1



def knn(k, digits_train, digits_test):
    """Uses K-NN to classify each of the test points
    k (int): the humber of nearest neighbors
    digits_train (list): The training points for each class
    digits_test (list): The testing points for each class
    returns (float): the error rate
    """
    predicted, actual = [], []
    for test_w, test_patterns in enumerate(digits_test):
        for test_x in test_patterns:
            distances = []
            for train_w, train_patterns in enumerate(digits_train):
                for train_x in train_patterns:
                    distances.append((sp.euclidean(test_x, train_x), train_w))
            predicted.append(classify_knn(k, distances))
            actual.append(test_w+1)

    cm = confusion_matrix(actual, predicted)
    w = sum([cm[i][j] for i in range(len(cm)) for j in range(len(cm[i])) \
             if i != j]) / len(actual)
    print("K={}, error={}\n".format(k,round(w, 3)), cm, end="\n\n")
    return w