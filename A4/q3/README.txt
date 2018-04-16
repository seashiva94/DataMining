#####################################
#				    #
#   Movie Ratings Predictor	    #	
#   	  	  		    #
#####################################

USAGE:
       ----------------------------------------------------------------------
       python recommender.py <training file> <test file> <distance> <k> <part>
       -----------------------------------------------------------------------
       training_file: if part = 1 or part = 2 training file can be ui.base [i = 1,2,3,4]
	 	      if part = 3 training file can be ri.train [i = 1,2,3,4]
	 test_file: if part = 1 or part = 2 test file can be ui.test [i = 1,2,3,4]
	 	    if part = 3 test file can be ri.test [i = 1,2,3,4]
	distance : [euc, manhattan, cosine]
	k: [integer value like 5,10,30,100...]
	part: [1,2,3] what part of the question
