import pandas as pd
import sys


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "Usage : python preprocess.py <filename>"
        
    filename = sys.argv[1]
    df = pd.read_csv(filename, header=None, sep = ",")
    df = pd.get_dummies(df)
    outfile = "data/" +filename.split(".")[0] + ".csv" 
    df.to_csv(outfile,index = None)
    
