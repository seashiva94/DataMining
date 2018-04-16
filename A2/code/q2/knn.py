import math
import sys
import numpy as np
import matplotlib
from scipy import spatial
from collections import Counter

matplotlib.use("agg")
import matplotlib.pyplot as plt

def euclidean(x,y):
    """
    computes the euclidean distance between two vectors X and Y
    """
    diff = x - y
    sq = diff**2
    total = np.sum(sq)
    return math.sqrt(total)

def cosine(x,y):
    """
    computes the cosine distance between vsctors X and Y
    """
    prod = np.dot(x,y)
    norm_x = math.sqrt(np.sum(x**2))
    norm_y = math.sqrt(np.sum(y**2))
    sim = prod / float(norm_x* norm_y)
    return 1-sim

def dist_3(x,y, p = 2):
    """
    computes distance between two vectors X and Y
    using equation (1) of the assignment
    """
    diff = x - y
    t1 = np.sum(diff[np.where(diff > 0)]) ** p
    t2 = np.abs(np.sum(diff[np.where(diff < 0)])) ** p
    return (t1 + t2)**(1/float(p))

def dist_4(x,y, p = 2):
    """
    computes distance between two vectors X and Y
    using equation (2) of the assignment
    """
    num = dist_3(x,y,p)
    total = 0
    diff = x - y
    for i in range(x.shape[0]):
        total += max(x[i],y[i], diff[i]) 

    return num / float(total)

def generate_data(n, k,method = "uniform"):
    """
    given n and k, generate n K-dimentional vectors
    using the distribution provided in method
    """
    if method == "uniform":
        x = np.random.uniform(0,1,(n, k))
        return x
    
    if method == "gaussian":
        cov = np.eye(k)
        mean = np.zeros(k)
        x = np.random.multivariate_normal(mean, cov, n)
        return x



def get_dist_counts(data, distance):
    """
    given a distance matrix, 
    return how many times
    each data point occurs
    """
    #print "in dist_counts"
    distances = spatial.distance.squareform(spatial.distance.pdist(data, distance))
    closest_5 = np.argpartition(distances, kth=5, axis=1)[:,:5]
    # no argpartition in np 1.7 on hulk
    counts = np.zeros(len(distances))
    #print closest_5
    
    for i in range(len(closest_5)):
        for j in range(5):
            if j != i:
                counts[closest_5[i][j]] += 1
            

   
    return counts

if __name__ == "__main__":
    Ks = [3,30,300,3000]
    methods =["uniform", "gaussian"]
    distances = [euclidean, cosine, dist_3, dist_4]
    N = 10000
    
    Ks = [3,30]
    #Ks = map(lambda x: int(x), sys.argv[1:])
    methods=["gaussian"]
    distances = [euclidean]#, cosine, dist1, dist2]
    for k in Ks:
        for method in methods:
            plt.figure()
            data = generate_data(N,k, method)
            for distance in distances:
                cnt = Counter()
                dist_counts = get_dist_counts(data, distance)
                #print dist_counts
                
                for d in dist_counts:
                    cnt[d] += 1
                    
                x = np.array(cnt.keys())
                y = np.array(cnt.values())
                y = y/float(np.sum(y))
                #print x
                # sort x and y as pairs
                x,y = zip(*sorted(zip(x,y), key=lambda x: x[0]))
                plt.plot(x,y, label = str(distance.__name__))
            plt.legend()
            plt.title("K = "+str(k)+", data = "+ str(method))
            plt.savefig(str(method) + "_"+str(k)+ ".pdf")
