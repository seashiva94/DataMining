###########################
# CURSE OF DOMENSIONALITY #
###########################


To run the code, execure the command, 

	python dimensionality.py

The code runs finds r(k) values for k=1 to k=100. It takes a long time to run for k=1000 and 10000 

* by deault the code uses gaussian distribution to generate data points
* by default the code runs for all distance metrics for N=100 and plots the graphs and saves them as pdf
* to run for a different value of N change the line #76 of the code
* to run for all 3 values of N uncomment line #77
* to run using uniform distribution to generate data points uncomment line #73
