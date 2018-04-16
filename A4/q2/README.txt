######################################
#				     #
#     Apriori Algorithm		     #
#				     #
######################################
------------------------------------------------------------------------------------
USAGE: python apriori.py <data_file> <min_sup> <min_conf> <methid> <implementation>
-----------------------------------------------------------------------------------
	 data_file : [data/cars.csv, data/mushrooms.csv, data/nursery.scv]
	 min_sup: minuimum support threshold: 0-1 
	 min_conf: minimum confidence threshold: 0-1 
	 method: [lift , conf] 
	 implementation: [prev, first] 
	 max_level: [integer 0,1,2,3,...] 


method: controls whether to use lift or confidence for generating rules
max_level: controls the maximum length of the itemset to be generated
impementation = prev: merge frequent k-1 itemsets to generate k candidates
impementation = prev: merge frequent k-1 itemsets and 1- itemsets to generate k candidates
