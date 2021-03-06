from __future__ import division
import math
import scipy as sp
from scipy import spatial
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd

def euclidean_dist(x,y):
    diff = x - y
    square = np.square(diff)
    dist = math.sqrt(np.sum(square))
    return dist

def minkowski_dist(x,y):
    p = 3.0
    abs_diff = np.abs(x - y)
    power = abs_diff**p
    total = np.sum(power)
    dist = (total) ** (1/p)
    return dist

def question_metric(x,y):

    p = 2
    x_greater_idx = np.where(x>y)[0]
    y_greater_idx = np.where(y>x)[0]
    s1 = np.sum((x[x_greater_idx] - y[x_greater_idx])**p)
    s2 = np.sum((y[y_greater_idx] - x[y_greater_idx])**p)
    dist = (s1+s2)**(1/float(p))
    return dist

def  citiblock_dist(x,y):
    diff = x - y
    abs_diff = np.abs(diff)
    dist = np.sum(abs_diff)
    return dist

def cosine_dist(x,y):
    mul = x.dot(y)
    norm_x = math.sqrt(np.sum(np.square(x)))
    norm_y = math.sqrt(np.sum(np.square(x)))
    cos = mul/(norm_x*norm_y)
    dist = 1-cos
    return dist


def generate_data(n,k, method="unif"):
    if method=="uniform":
        x = []
        for i in range(n):
            x.append(np.random.uniform(0,1,k))
    else:
        cov = np.eye(k)
        mean = np.zeros(k)
        x = np.random.multivariate_normal(mean, cov, n)
    return np.array(x)

def find_r(data, func):
    dists = spatial.distance.pdist(data, func)
    r = (max(dists) - min(dists)) / min(dists)
    if r <= 0:
        return 0
    
    return math.log10(r)


if __name__ == "__main__":
   
    method = "gaussian"
    #method = "uniform"
    funcs = [euclidean_dist, minkowski_dist, question_metric, citiblock_dist, cosine_dist]

    N = [100]
    #N = [100,1000,10000]
    
    K = range(1,101)
    for func in funcs:
       
        R = []
        for n in N:
            r = []
            for k in K:
                total = 0
                for i in range(5):
                    data = generate_data(n,k, method)
                    total += find_r(data, func)
                total = total/5
                r.append(total)
                print func.__name__, n, k, total
            R.append(r)
           
        plt.figure()

        print "plotting"
        for j in range(len(N)):
            plt.plot(K,R[j], label="n = "+ str(N[j]))
            
        plt.legend()
        plt.title(func.__name__ + " with " + method + " data")
        plt.savefig(func.__name__ + "_" + method + ".pdf")
        print "plotted"
    
