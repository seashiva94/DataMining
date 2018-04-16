import pandas as pd
import sys

filename = sys.argv[1]
sep = ','
if filename.endswith("birch.txt"):
    infile = open("birch.txt", "r")
    outfile = open("processed_birch.csv", "w")
    i = 0
    for line in infile:
        l = line.split(" ")
        t = []
        for num in l:
            if num != "":
              t.append(num.strip())  
        print str(i) + "," +t[0]+","+t[1]+"\n"
        outfile.write(str(i) + "," +t[0]+","+t[1]+"\n")
        i+= 1
    exit()
df = pd.read_csv(filename, sep, header=None)
n = df.shape[1] -1

if filename.lower().startswith("wine"):
    columns = df.columns.tolist()
    columns = columns[1:] + [columns[0]]
    df = df[columns]
    df[n] = df[n].astype("category").cat.codes
    
if filename.lower().startswith("magic"):
    pass

if filename.lower().startswith("breast"):
    #df.drop(0, axis=1)
    #n = df.shape[1] -1
    columns = df.columns.tolist()
    columns = [columns[0]]+ columns[2:] + [columns[1]]
    df = df[columns]
    df[1] = df[1].astype("category").cat.codes
#print df.head()

print df.dtypes

df.to_csv("processed_" +filename,header=False)
df2= pd.read_csv("processed_" +filename, header=None,index_col=0)
#print df2.head()
print df2.dtypes

