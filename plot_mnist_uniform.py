"""
This file simply generates a count of the datapoints for each class, and plots
them on two bar graphs.

Author: Brenden Hein
"""

import matplotlib.pyplot as plt


def plot(data, name):
    """Plots the histogram of the dataset
    data (list): The patterns in the dataset
    name (str): Training or Testing
    """
    fig, ax = plt.subplots()
    classes = [i for i in range(len(data))]
    count = [len(d) for d in data]
    ax.bar(classes, count)
    ax.set_xlabel("digit")
    ax.set_ylabel("frequency")
    ax.set_title(name + " Class Distribution in MNIST")
    

def read_file(fp, name):
    """Reads the file to extract the datat for the patterns
    fp (file pointer): the opened file
    name (str): Training or Testing
    """
    digits = [[], [], [], [], [], [], [], [], [], []]
    for pattern in fp:
        pattern = pattern.strip().split(",")
        w, p = int(pattern[-1]), [int(p) for p in pattern[:-1]]
        digits[w].append(p)
    plot(digits, name)
    

def main():
    read_file(open("features_train.csv"), "Training")
    read_file(open("features_test.csv"), "Testing")
    
if __name__ == "__main__":
    main()