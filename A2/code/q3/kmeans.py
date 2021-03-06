import sys
import numpy as np
import pandas as pd

def manhattan(x,y):
    return np.sum(np.abs(x-y))

def euclidean(x,y):
    return np.sqrt(np.sum((x-y)**2))

def cosine(x,y):
    """                                                                                                                                              
    computes the cosine distance between vsctors X and Y
    """
    prod = np.dot(x,y)
    norm_x = np.sqrt(np.sum(x**2))
    norm_y = np.sqrt(np.sum(y**2))
    sim = prod / float(norm_x* norm_y)
    return 1-sim
                
def dist_3(x,y, p=2):
    """
    computes distance of two vectors x and y
    using equation(1) of the assignment
    """
    diff = x - y
    t1 = np.sum(diff[np.where(diff > 0)]) ** p
    t2 = np.abs(np.sum(diff[np.where(diff < 0)])) ** p
    return (t1 + t2)**(1/float(p))
            
    
def dist_4(x,y, p=2):
    """
    computes distance of two vectors x and y
    using equation(2) of the assignment
    """
    num = dist_3(x,y,p)
    total = 0
    diff = x - y
    for i in range(x.shape[0]):
        total += max(x[i],y[i], diff[i])

    return num / float(total)
                            

class KMeans(object):
    def __init__(self, k, distance):
        self.num_clusters = k
        self.distance = distance
        self.iters = 0
        self.dist_computations = 0
        self.sse = 0

    def find_closest_centroid(self, x):
        """
        returns index of closest centroid
        """
        dists = np.zeros(self.num_clusters)
        for i in range(self.num_clusters):
            dists[i] = self.distance(x,self.centroids[i])
            self.dist_computations +=1
        closest_centroid_idx = np.argmin(dists)
        return closest_centroid_idx

    def update_centroids(self, data):
        """
        performs one iteration of Kmeans, and updates the centroids
        """
        new_centroids = np.zeros_like(self.centroids)
        counts = np.zeros(self.num_clusters)
        for x in data:
            idx = self.find_closest_centroid(x)
            new_centroids[idx] += x
            counts[idx] += 1
            
        #print "new cent: ", new_centroids, new_centroids.shape
        #print "counts:", counts, counts.shape
        for i in range(new_centroids.shape[0]):
            new_centroids[i] = new_centroids[i]/counts[i]
        #new_centroids = new_centroids/counts
        
        if np.all(new_centroids == self.centroids):
            self.changing = False
        else:
            self.centroids = new_centroids
    

    def find_sse(self, data):
        """
        finds the sum of squared errors
        given for a given data
        """
        predicted = self.assign_cluster(data)
        for i in range(len(predicted)):
            self.sse += self.distance(data[i], self.centroids[predicted[i]])**2
            
    def cluster_data(self,data):
        """
        Find centroids of data using Kmeans
        """
        centroids_idx = np.random.choice(data.shape[0],self.num_clusters, replace=False)
        self.centroids = data[centroids_idx]
        self.changing = True
        while self.changing:
            self.iters += 1
            print "iters = ", self.iters
            self.update_centroids(data)
        self.find_sse(data)

    def assign_cluster(self, data):
        """
        data is a numpy array
        for each data point returns a cluster assignment
        between 0 and k-1
        """
        assignments = []
        for x in data:
            c = self.find_closest_centroid(x)
            assignments.append(c)
        return assignments

    def get_rand_index(self, data, labels):
        """
        given a set of training data and labels
        learn centroids
        evaluate performance based on external index
        according to rand index
        [https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html]
        """
        self.cluster_data(data)
        predicted = self.assign_cluster(data)
        print "pred = ", predicted
        print "actual = ", labels
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        RI = 0

        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j:
                    if labels[i] == labels[j]:
                        if predicted[i] == predicted[j]:
                            tp += 1
                        else:
                            fn += 1
                    else:
                        if predicted[i] == predicted[j]:
                            fp += 1
                        else:
                            tn += 1
        RI = (tp + tn)/float(tp + fp + fn + tn)
        return RI

    def evaluate_clustering(self, data, labels):
        # repeat 5 times
        # self.get rand_index
        # find the set of centroids with best rand index
        # 
        pass

class ElkansKMeans(KMeans):

    def find_s(self):
        """
        for all centroids return distance to closest centroid/2
        """
        print self.centroids
        s = np.zeros(self.num_clusters)
        for i in range(self.num_clusters):
            dists = []
            for j in range(self.num_clusters):
                if i !=j:
                    dists.append(self.distance(self.centroids[i], self.centroids[j]))
            s[i] = np.min(dists)
        #print "s is:", s
        return s/float(2)

    
    def update_centroids(self, data):
        s = self.find_s()
        change_points_idx = np.where(self.u > s[self.a])[0]
        new_centroids = np.zeros_like(self.centroids)
        counts = np.zeros(self.num_clusters)

        """
        print "U: ", self.u
        print "L: ", self.l
        print "A: ", self.a
        """
        
        for i in change_points_idx:
            for j in range(self.num_clusters):
                self.r = True
                z = max(self.l[i][j], self.distance(self.centroids[self.a[i]], self.centroids[j])/float(2))
                if ((self.a[i] != j) and (self.u[i] > z) and (self.r)):
                    self.u[i] = self.distance(data[i], self.centroids[self.a[i]])
                    self.dist_computations += 1
                    self.r = False

                   
                    
                if self.u[i] > z:
                    self.l[i][j] = self.distance(data[i], self.centroids[j])
                    self.dist_computations += 1

                if self.l[i][j] < self.u[i]:
                    self.a[i] = j
                    new_centroids[j] += data[i]
                    counts[j] += 1
                    

        for i in range(len(new_centroids)):
            if counts[i] == 0:
                new_centroids[i] = self.centroids[i]
            else:
                new_centroids[i] = new_centroids[i]/counts[i]
        
        delta = np.zeros(self.num_clusters)
        for i in range(data.shape[0]):
            for j in range(self.num_clusters):
                delta[j] = self.distance(self.centroids[j], new_centroids[j])
                self.u[i] += delta[self.a[i]]
                self.l[i][j] += max(0, self.l[i][j] - delta[j])
        
        if np.all(self.centroids == new_centroids):
            self.changing = False
        else:
            if np.sum(counts) > 0:
                self.centroids = new_centroids


    def cluster_data(self, data):
        self.a = np.zeros(data.shape[0]).astype(int) # assignment
        self.l = np.zeros((data.shape[0], self.num_clusters)) # lower bounds to each centroid
        self.u = np.full(data.shape[0], np.inf) # upper bound to current centroid
        self.r = True
        super(ElkansKMeans, self).cluster_data(data)

    
if __name__ == "__main__":

    if len(sys.argv) != 5:
        print len(sys.argv)
        print "USAGE:"
        print sys.argv[0], "<filename>","<method>", "<distance>", "<k>"
        print "filename : [data/iris_data.csv, data/wine_data.csv]"
        print "method :  [kmeans, elkans]"
        print "distance : [euclidean, cosine, mahattan, dist_3, dist_4]"
        print "k : the number of clusters needed"
        exit()

    distances = {"euclidean": euclidean,
                 "cosine": cosine,
                 "manhattan": manhattan,
                 "dist_3": dist_3,
                 "dist_4": dist_4}
    
    filename = sys.argv[1]
    method = sys.argv[2]
    distance = distances[sys.argv[3]]
    k = int(sys.argv[4])
    data_df = pd.read_csv(filename, header=None, index_col=0)
    
    print(data_df.head())
    """
    if filename.endswith("birch.csv"):
        print "here"
        data = data_df.as_matrix()
        print data[0]
        kmeans_clustering = KMeans(k=k, distance=distance)
        elkans_clustering = ElkansKMeans(k=k, distance=distance)
        print "running kmeans"
        kmeans_clustering.cluster_data(data)
        print "computations using kmeans :", kmeans_clustering.dist_computations
        print "iters :", kmeans_clustering.iters
        print "sse :" ,kmeans_clustering.sse
        print "distance computatuins", kmeans_clustering.dist_computations
        
        print "running elkans kmeans"
        elkans_clustering.cluster_data(data)
        print "computations using elkans :", elkans_clustering.dist_computations
        print "iters : ", elkans_clustering.iters
        print "sse :" ,elkans_clustering.sse
        print "distance computatuins", elkans_clustering.dist_computations
        exit()
    """
    data = data_df.as_matrix()[:,:-1]
    labels = data_df.as_matrix()[:,-1]
    if method == "kmeans":
        kmeans_clustering = KMeans(k=k, distance= distance)
    elif method == "elkans":
        kmeans_clustering = KMeans(k=k, distance= distance)
    else:
        print "invalid method"
        print "method : [kmeans, elkans]"
        exit()

    #kmeans_clustering.cluster_(data, labels)
    ri = kmeans_clustering.get_rand_index(data, labels)
    print "assignments: ", kmeans_clustering.assign_cluster(data)
    print "computations", kmeans_clustering.dist_computations
    print "iters", kmeans_clustering.iters
    print "sse: ", kmeans_clustering.sse
    print "rand Index is", ri
