import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ks= [3,30,300,3000]
    methods = ["uniform", "gaussian"]
    distances = ["Euclidean", "Cosine", "Distance 3", "Distance 4"]

    for method in methods:
        for k in ks:
            plt.figure()
            for distance in distances:
                data = np.loadtxt("plot_data/" +
                                  distance + "_" + str(k) + "_"+
                                  method + ".txt", delimiter=',')
                legend = "_".join(distance.lower().split(" "))
                plt.plot(data[0], np.log(data[1]), label = legend)
            plt.legend()
            plt.xlabel("N5")
            plt.ylabel("Frequency")
            plt.title("K = "+str(k)+","+"Data = "+ method)
            plt.savefig("semilog"+method+ "_"+ str(k) +".pdf")