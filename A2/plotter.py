import xlrd
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_excel("R_gaussian.xlsx")
df2 = pd.read_excel("R_uniform.xlsx")


#print df1.head()
#print df2.head()
"""
df_100 = df1[df1['n']==100]
df_1000 = df1[df1['n']==1000]
df_10000 = df1[df1['n']==1000]

print df_100
names = set(df_100['distance'])
"""
dist_grp= df1.groupby(['distance']).groups
print dist_grp
for d, idx in dist_grp.iteritems():
    plt.figure()
    df_o = df1.iloc[idx,:]
    #print df_o
    grp2 = df_o.groupby('n').groups
    #print grp2
    #exit()
    for n, idx in grp2.iteritems():
        data = list(df1.iloc[idx,:]['val'])
        k = list(df1.iloc[idx,:]['k'])
        plt.plot(k, data, label="n="+str(n))
    
    plt.xlabel("K")
    plt.ylabel("r(K)")
    plt.legend()
    plt.savefig(d+'_norm.pdf')
