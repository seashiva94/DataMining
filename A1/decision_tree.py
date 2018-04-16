import numpy as np
import pandas as pd
import math
import sys

class Node:
    threshold = -1
    feature_idx = -1
    data_idxs = [] 
    left = None
    right = None
    label = None
    

def tree_string(root):
    temp = root
    # root-left-right
    return 

def stopping_condition(data):
    stop = False 
    if data.shape[0] <= 10:
        stop = True
        
    groups = data.groupby(data.shape[1] -1)
    if groups.ngroups == 1:
        stop = True
        
    return stop

def get_gini(data):    
    gini = 0
    n = data.shape[0]
    groups = data.groupby(data.shape[1] - 1).groups
    total = 0
    for k, v in groups.iteritems():
        total += (len(v)/float(n))**2
        
    gini = 1 - total
    return gini

def get_entropy(data):
    entropy = 0
    n = data.shape[0]
    groups = data.groupby(data.shape[1] - 1).groups
    for k, v in groups.iteritems():
        p = len(v)/float(n)
        entropy += -p*math.log(p,2)
    return entropy

def get_metric(data,feature, threshold, method = "GINI"):

    n = data.shape[0] 
    left_data = data.loc[data[feature] < threshold]
    right_data = data.loc[data[feature] > threshold]
    if method == "GINI":
        func = get_gini
    else:
        func = get_entropy
        
    before = func(data)
    after = left_data.shape[0]/float(n) * func(left_data) + right_data.shape[0]/float(n) *func(right_data)
    gain = before - after
    return gain
    

def train(data, method = "GINI", node="root"):
    #print node
    print method
    root = Node()

    if stopping_condition(data):
        grp = data.groupby(data.shape[1] - 1)
        groups = grp.groups
        max_group = 0
        max_count = 0
        for k, v in groups.iteritems():
            if len(v) > max_count:
                max_group = k
                
        root.label = max_group
        return root

    else:       
        best_feature = 0
        best_feature_thresh = 0
        best_feature_metric = -float('INF')
        for i in range(data.shape[1] -1):
            temp = data.sort_values(by = i)
            thresh = 0
            metric = -float('INF')
            for j in range(1, data.shape[0]):
                if data.iloc[j,-1] != data.iloc[j-1,-1]:
                    temp = (data.iloc[j,i] + data.iloc[j-1,i])/float(2)
                    temp_metric =  get_metric(data, i, temp, method)
                    
                    if temp_metric > metric:
                        thresh = temp
                        metric = temp_metric
            if metric > best_feature_metric:
                best_feature_metric = metric
                best_feature_thresh = thresh
                best_feature = i

        
        left_data = data.loc[data[best_feature] < best_feature_thresh]
        right_data = data.loc[data[best_feature] > best_feature_thresh]


        root.feature_idx = best_feature
        root.threshold = best_feature_thresh
        root.left = train(left_data, method, "left")
        root.right = train(right_data, method, "right")
        
        return root

def classify(x, root):
    label = None
    temp = root
    while temp != None:
        label = temp.label
        if x[temp.feature_idx] > temp.threshold:
            temp = temp.right
        else:
            temp = temp.left
    return label

def test_performance(data, method):
    
    n = len(data)
    data = np.array(data)
    total = 0
    best_acc = 0
    best_root = None
    for i in range(5):
        idxs = np.random.permutation(n)
        training_data = data[idxs[0:n/2]]
        test_data = data[idxs[n/2:]]

        training_data_df = pd.DataFrame(training_data)
        root = train(training_data_df, method,node ="root")
        print root.__dict__
        predicted = [classify(x, root) for x in test_data[:,:-1]]
        predicted = np.array(predicted)
        actual_labels = test_data[:,-1]

        correct_count = len(np.where(predicted == actual_labels)[0])
        accuracy = correct_count/float(test_data.shape[0])

        if accuracy > best_acc:
            best_acc = accuracy
            best_root = root
        total += accuracy
    average = total / float(5)

    return best_root, average

if __name__ == "__main__":


    if len(sys.argv) < 2:
        print "ERROR !! INCORRECT USAGE"
        print "USAGE: "
        print "python decision_tree.py <method> <filename>"
        print "method : [GINI, ENTROPY]"
        print "filename: [Iris_Data.csv, BreastCancer_Data.csv, Wine_Data.csv]"
        exit()
    method = sys.argv[1]
    filename = sys.argv[2]
    data_df = pd.read_csv(filename, header=None, index_col=0)
    best_root, avg_accuracy = test_performance(data_df, method)
    
    print avg_accuracy
    tree_string(best_root)
    print best_root.__dict__
