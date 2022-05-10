"""
This is the main file responsible for classifying a point.  It uses KNN and 
Parzen window to classify, MDA to transform features, and also handles
plotting.

Author: Brenden Hein
"""

import knn
import parzen
import matplotlib.pyplot as plt
import numpy as np
import itertools

FEATURES = ["Upper Left Boundary", "Middle Upper Boundary",  \
          "Upper Right Boundary", "Middle Right Boundary",
          "Lower Right Boundary", "Middle Lower Boundary",
          "Lower Left Boundary", "Middle Left Boundary"]
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    

def error_data(errors, method):
    """Handles deriving pertinent information about the different error
    rates for the subsets of the MINST traingin and testing datasets
    errors (list): The error rates for each of the subsets
    method (str): K-NN or Parzen Window
    """
    emean = sum(errors)/len(errors)
    print("Error Mean: ", emean)
    print("Error Variance: ", np.var(errors))
    
    fig, ax = plt.subplots()
    ax.scatter([i for i in range(1, 10)], errors)
    ax.hlines(emean, 0, 10, linestyles="dashed")
    ax.set_xlabel("Subset Number")
    ax.set_ylabel("Error Rate")
    ax.set_title("Error Rates for MNIST subsets using {}".format(method))


def plot_MDA(digits, E, mu, set_type):
    """Plots the best 2 features using PCA feature transformation
    digits (list): A subset containing the patterns for each digit
    E (list): The top eigen vectors for the digits features
    mu (list): The overall mean of all the patterns, independent of class
    set_type (str): Training or Testing
    returns (list): The transformed data points
    """
    fig1, ax1 = plt.subplots()
    digits_mda = []
    for w, features in enumerate(digits):
        x1, x2, ys = [], [], []
        for f in features:
            y = (E.T).dot(f-mu)  # Transforms a point to 2 dimensions
            x1.append(y[0])
            x2.append(y[1])
            ys.append(y)
        ax1.scatter(x1, x2, label=CLASSES[w], alpha=.4)
        digits_mda.append(ys)
    
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title("MDA Feature Transformation for MNIST ({})".format(set_type))
    ax1.legend()
    
    return digits_mda


def plot_feature_histo(feat, feature_data):
    """This function simply plots a histograms of hte 10 classes [0-9] to get 
    a visual feel of the seperation of classes for EACH feature
    feat (str): The feature name
    feature_data (list): the feature data for each of the 10 classes
    """
    fig, ax = plt.subplots()
    for i, w in enumerate(feature_data):
        ax.hist(w, label=i, density=True)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")
    ax.set_title(feat)
    ax.legend()
        

def MDA(train, test, size):
    """This function applies MDA to reduce the features necessary for
    classification in the form of feature transformation
    train (list): the training patterns for each class
    test (list): the testing patterns for each class
    size (int): the subset size
    """
    # Combines the train and test features into one feature set
    digits = []
    for i in range(len(train)):
        digits.append(train[i] + test[i])
    
    # Finds the overall mean of the digits
    digits_all = list(itertools.chain.from_iterable(digits))
    mu_o = np.mean(digits_all, axis=0)
    
    # Calculates SW, the scatter matrices WITHIN classes and BETWEEN classes
    SW, mus = [], []
    for w, features in enumerate(digits):
        mu = np.mean(features, axis=0)
        SW.append(sum([np.outer((el-mu), (el-mu)) for el in features]))
        mus.append(mu)
    SW = sum(SW)   
    SB = sum([len(digits[i])*(np.outer((mus[i]-mu_o), (mus[i]-mu_o))) \
              for i in range(len(digits))])
    
    # Picks the top eigen vectors besed on their eigen values
    S = np.dot(np.linalg.inv(SW), SB)
    S_eigval, S_eigvec = np.linalg.eigh(S)
    indices = (-S_eigval).argsort()[:2]
    E = S_eigvec[:, [indices[0], indices[1]]]
    
    # Splits the training and testing sets back up once the directions 
    # of maximum variance are founds
    train, test = [], []
    for patts in digits:
        train.append(patts[:size])
        test.append(patts[size:])
    digits_mda_train = plot_MDA(train, E, mu_o, "Training") 
    digits_mda_test = plot_MDA(test, E, mu_o, "Testing")
    return digits_mda_train, digits_mda_test


def get_data(filename, max_size, subset_size):
    """Given a filename, it parses the file to extract the classes/features
    then organizes that data into subsets, with each subset
    a portion of the data for each class.
    filename (str): the name of the file to open
    max_size (int): The total number of datapoints
    subset_size (int): How big each of the subsets of patterns will be
    return (list): The data gathered from the file
    """
    fp = open(filename)
    
    # Creates are list of all subsets of the training and testing data
    digits, current = [], [0]*10
    bins = max_size//(10*subset_size)-1
    for i in range(bins):
        digits.append([[], [], [], [], [], [], [], [], [], []])
        
    for pattern in fp:
        pattern = pattern.strip().split(",")
        w, p = int(pattern[-1]), [int(p) for p in pattern[:-1]]
        
        # Takes a subset of the data
        if current[w] < bins and len(digits[current[w]][w]) < subset_size: 
            digits[current[w]][w].append(p)
        else:
            current[w] += 1
            
    fp.close()       
    return digits
    

def get_all_data(filename):
    """Given a filename, it parses the file to extract the classes/features
    then organizes that data into a SINGLE set of the 10 classes
    filename (str): the name of the file to open
    return (list): The data gathered from the file
    """
    fp = open(filename)
    digits = [[], [], [], [], [], [], [], [], [], []]
    for pattern in fp:
       pattern = pattern.strip().split(",")
       w, p = int(pattern[-1]), [int(p) for p in pattern[:-1]]
       digits[w].append(p)
       
    fp.close()
    return digits


def main():
    size = int(input("Enter a subset size: "))
    
    # Parses the feature file 
    digits_train = get_data("features_train.csv", 60000, size)
    digits_test = get_data("features_test.csv", 10000, size//6)
    digits_train_all = get_all_data("features_train.csv")    
    digits_test_all = get_all_data("features_test.csv")
    
    # Gets a list[lists], with each list containing all of "feat" for a class
    if input("Plot histogram of IMOX features (y/n): ").lower() == "y":
        for i, feat in enumerate(FEATURES):
            feature_data = []
            for d in digits_train_all:
                feature_data.append([f[i] for f in d])
            plot_feature_histo(feat, feature_data)
    
    # Uses MDA to transform the features into 2 dimensions
    if input("Use MDA for feature tranformation (y/n): ").lower() == "y":
        subnum = int(input("Enter a subset number from 0-{}: ".format(len(digits_train)-1)))
        MDA_train, MDA_test = MDA(digits_train[subnum], digits_test[subnum], size)
        knn.knn(int(input("Input a K: ")), MDA_train, MDA_test)
        parzen.parzen(MDA_train, MDA_test, float(input("Window Size: ")), 2)
    
    # Uses K-NN to classify each each point to the best of its ability
    if input("Use KNN (y/n): ").lower() == "y":
        if input("Apply to all samples (y/n): ").lower() == "y":
            knn.knn(int(input("Input a K: ")), digits_train_all, digits_test_all)
        else:
            k_in = input("How many neighbors (start,end,step): ").split(",")
            k_start, k_end, step = int(k_in[0]), int(k_in[1]), int(k_in[2])
            
            # Loops through all values of k
            for k in range(k_start, k_end, step):
                errors = []
                for i in range(len(digits_train)):
                    err = knn.knn(k, digits_train[i], digits_test[i])
                    errors.append(err)
                error_data(errors, "K-NN: K={}".format(k))
        
    # Uses a parzan window to classify each point to the best of its ability
    if input("Use Parzen (y/n): ").lower() == "y":
        if input("Apply to all samples (y/n): ").lower() == "y":
            parzen.parzen(digits_train_all, digits_test_all, float(input("Window size: ")), 8)
        else:
            # Loops through all values of h
            for h in [.1, .25, .5, 1, 1.5, 2]:
                errors = []
                for i in range(len(digits_train)):
                    err = parzen.parzen(digits_train[i], digits_test[i], h, 8)
                    errors.append(err)
                error_data(errors, "Parzen Windows: h={}".format(h))
                
if __name__ == "__main__":
    main()