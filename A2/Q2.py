#!/usr/bin/env python3
"""
@author: Pulkit Maloo
"""
from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter
import matplotlib.pyplot as plt
import time

# Use Libraries, Set to 3 to run on server
method = 1


def generate_data(n, k, kind="uniform"):
    """ Generate data of n x k dimension of kind uniform or gaussian """
    if kind == "uniform":
        return np.random.rand(n, k)
    elif kind == "gaussian":
        return np.random.multivariate_normal(mean=np.zeros(k),
                                             cov=np.identity(k), size=n)
    else:
        raise ValueError("Invalid kind of data")


def distance3(p1, p2, p=2):
    """ p1 and p2 should be k-dim data points and not nxk dim vectors"""
    diff = p1-p2
#    f_plus = np.vectorize(lambda x: max(x, 0))
#    f_minus = np.vectorize(lambda x: max(-x, 0))
#    return np.power(np.sum((diff).clip(0))**p +
#                    np.sum((-diff).clip(0))**p, 1/p)
    return np.power(np.sum((diff).clip(0), axis=1, keepdims=True)**p +
                    np.sum((-diff).clip(0), axis=1, keepdims=True)**p, 1/p)


def distance4(p1, p2, p=2):
    # numerator
    num = distance3(p1, p2)
    # denominator
    den = np.sum(np.maximum(np.abs(p1), np.abs(p2), np.abs(p1-p2)),
                 axis=1, keepdims=True)
    return num/den


def nearest_neighbors(data, metric):
    global method
    tic = time.time()
    if isinstance(metric, str):
        if method == 1:
            from sklearn.neighbors import NearestNeighbors
            model = NearestNeighbors(n_neighbors=5, metric=metric)
            model.fit(data)
            neigh = model.kneighbors(return_distance=False)

        elif method == 2:
            from sklearn.metrics.pairwise import pairwise_distances
            dist_matrix = pairwise_distances(data, data, metric)
            neigh = dist_matrix.argsort()[:, 1: 6]

        elif method == 3:
            dist_matrix = cdist(data, data, metric)
            neigh = dist_matrix.argsort()[:, 1: 6]
    elif callable(metric):
        dist_matrix = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]):
            dist_matrix[i, :] = metric(data, data[i].reshape(1,
                                       data.shape[1])).reshape(-1)
        neigh = dist_matrix.argsort()[:, 1: 6]

    else:
        raise ValueError("Function for metric is not found", metric)

    toc = time.time()
    print(round(toc-tic, 2), "seconds elapsed")
    return neigh


def compute_N5(data, metric):
    neigh = nearest_neighbors(data, metric)
    neigh_counts = Counter(neigh.reshape(-1))
    N5 = [neigh_counts[i] for i in range(data.shape[0])]
    return N5


def plot_N5(N5_dict, metrics, k, kind):

    for metric in metrics:
        legend = {"euclidean": "Euclidean",
                  "cosine": "Cosine",
                  distance3: "Distance 3",
                  distance4: "Distance 4"}[metric]

#        plt.hist(N5_dict[metric])
#        sns.kdeplot(np.array(N5_dict[metric]), label=legend, gridsize=1000)
        N5_count = dict(sorted(dict(Counter(N5_dict[metric])).items()))
        #plt.plot(N5_count.keys(),
        #         np.array(list(N5_count.values()))/sum(N5_count.values()),
        #         label=legend)

        #create a numpy array and save to file
        x =  np.array([N5_count.keys(), np.array(list(N5_count.values()))/sum(N5_count.values())])
        np.savetxt(legend + "_" +str(k) + "_" + kind + ".txt", x, delimiter=",")
        #exit()
    """    
    plt.xlabel("Value of n5")
    plt.ylabel("Fraction")
    plt.title("Plot of n5 for k="+str(k))
    plt.legend(loc="best")
    plt.savefig(fname=kind+"_"+str(k))
    plt.show()
    """

def main(kind):
    n = 10000
    k_range = [3, 30, 300, 3000]
    metrics = ("euclidean", "cosine", distance3, distance4)
    for k in k_range:

        tic = time.time()

        N5_dict = dict()

        data = generate_data(n, k, kind)

        for metric in metrics:

            print("Computing N5 for k =", k, "and metric:", metric)
            N5 = compute_N5(data, metric)

            N5_dict[metric] = N5

        toc = time.time()
        print("Total", round((toc-tic)/60, 2), "minutes elapsed")

        plot_N5(N5_dict, metrics, k, kind)


if __name__ == '__main__':
    tic = time.time()
    #main(kind="uniform")
    main(kind="gaussian")
    tac = time.time()
    print("Total", round((tac-tic)/60, 2), "minutes elapsed")
