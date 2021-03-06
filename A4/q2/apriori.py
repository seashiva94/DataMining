import sys
import numpy as np
import pandas as pd
from copy import deepcopy

# add k-1 itemsets and k-1 itemsets to generate k-itemses
class Apriori1:

    def __init__(self, min_sup = 0.20, min_conf = 0.1, max_level = 3):
        """
        initialize values
        """
        self.min_sup = min_sup
        self.min_conf = min_conf
        self.max_level = max_level
        self.freq_item_sets = {} # {level: itemsets}
        self.candidates  = {} # {level, itemsets}
        self.support_counts = {} #{str(itemset), support_count}
        self.interesting_rules = [] # [(a,b)] : {a} -> {b}
        self.rule_confidence = [] # [0.6,0.44, ...]
        self.maximal_freq = {}
        self.closed_freq = {}
    
    def generate_candidates(self, data, level = 1):
        """
        generate candidate itemsets for a given level
        data is a numpy matrix, with binary values
        1 for occurence of an item 
        0 for non occurence of the item
        """
        
        size = data.shape[0]
        
        if level == 1:
            candidates = np.arange(data.shape[1])
            candidates = [[c] for c in candidates]
            self.candidates[level] = [candidate for candidate in candidates]
            return candidates
        
        else:
            prev_frequent = self.freq_item_sets[level-1]
            candidates = []
            for c_1 in prev_frequent:
                for c_2 in prev_frequent:
                    if c_1 != c_2:
                        #can = list(np.unique(np.sort(can)))
                        if c_1[:-1] != c_2[:-1]:
                            continue
                        can = sorted(c_1 + [c_2[-1]])
                        if can in candidates:
                            continue
                        candidates.append(can)

            self.candidates[level] = candidates
            return candidates

    def find_frequent(self, data, level = 1):
        """
        find frequent itemsets for a gven level
        """
        check = True
        is_freq = False
        prev_freq = []
        self.size = data.shape[0]
        frequent = []
        candidates = self.candidates[level]
        if level == 1:
            check = False
        else:
            prev_freq = self.freq_item_sets[level-1]
        for candidate in candidates:
            temp = data[:,candidate]
            rows = np.where(np.all(temp == 1, axis = 1))[0]
            support =  len(rows)
            self.support_counts[str(candidate)] = support

            if support > self.min_sup*self.size:
                frequent.append(candidate)
                is_freq = True

            if check:
                c_1 = candidate[:-1]
                c_2 = candidate[:-2] + [candidate[-1]]
                #print candidate , c_1, c_2
                if len(c_1) != 0:
                    support_c1 = self.support_counts[str(c_1)]
                else:
                    support_c1 = 0
                if len(c_2) != 0:
                    support_c2 = self.support_counts[str(c_2)]
                else:
                    support_c2 = 0
                if is_freq:
                    self.maximal_freq[str(c_1)] = 1
                    self.maximal_freq[str(c_2)] = 1
                else:
                    print "here"
                    self.maximal_freq[str(c_1)] = 0
                    self.maximal_freq[str(c_2)] = 0
                
                if support_c1 != support:
                    if str(c_1) not in self.closed_freq:
                        self.closed_freq[str(c_1)] = 0
                else:
                    self.closed_freq[str(c_1)] = 1

                if support_c2 != support:
                    if str(c_1) not in self.closed_freq:
                        self.closed_freq[str(c_1)] = 0
                else:
                    self.closed_freq[str(c_1)] = 1
                
        self.freq_item_sets[level] = frequent
        return frequent

    def apriori_frequent_itemsets(self, data):
        """
        use apriori property to find frequent itemsets upto max_level
        """
        self.size = data.shape[0]
        #print "self.size", self.size
        for level in range(1,self. max_level+1, 1):
            candidates = self.generate_candidates(data, level)
            freq = self.find_frequent(data, level)
            
            print "at level: ", level
            print "num candidates: ", len(candidates)
            print "num frequent: ", len(freq)
            if len(freq) == 1:
                # only one frequent item set remaining
                # no need to go to next level... break
                break

    def is_interesting(self, rule, method = "conf", size=1):
        """
        rule : a touple, antecedent and consequent
        method : "conf" for conficdence, "lift for lift"
        """
        if method == "conf":
            confidence = self.support_counts[str(sorted(rule[0]+rule[1]))] *1.0
            confidence /= self.support_counts[str(rule[0])]
            truth  =  confidence >= self.min_conf
            return truth, confidence
        
        elif method == "lift":
            confidence = self.support_counts[str(sorted(rule[0]+rule[1]))] *1.0
            confidence /= self.support_counts[str(rule[0])]

            sup = self.support_counts[str(rule[0])]/float(self.size)
            lift = confidence/float(sup)
            truth = confidence >= self.min_conf
            return truth, lift
        else:
            print "Invalid measure of intersting-ness"
            exit()
            return

    def generate_rules(self, rule, method):
        
        if len(rule[0]) == 1:
            return
        
        for item in rule[0]:
            temp_rule = deepcopy(rule)
            temp_rule[1].append(item)
            temp_rule[1].sort()
            temp_rule[0].remove(item)
            truth, confidence = self.is_interesting(temp_rule, method = method)
            if truth:
                if temp_rule not in self.interesting_rules:
                    self.interesting_rules.append(temp_rule)
                    self.rule_confidence.append(confidence)
                self.generate_rules(temp_rule, method)

        return

    def write_rules_to_file(self,filename, columns):
        with open(filename, 'w') as outfile:
            #ctr = 0
            # sort based on confidence
            # write using header info

            confident_rule_idx = np.argsort(self.rule_confidence)[-5:]
            for idx in confident_rule_idx:
                rule = self.interesting_rules[idx]
                string = "if "
                for item in rule[0][:-1]:
                    string += columns[item] + " and "
                string += columns[rule[0][-1]]
                string += " ===> "
                for item in rule[1][:-1]:
                    string += columns[item] + " and "
                string += columns[rule[1][-1]]
                string += "\t confidence = " + str(self.rule_confidence[idx]) + "\n"
                #string = str(rule[0]) +  " ===> " + str(rule[1])
                #string += "\t confidence = " + str(self.rule_confidence[ctr])
                #string += "\n"
                outfile.write(string)
                print string
                #ctr += 1
        
    def find_interesting_rules(self, method = "lift", columns = [], filename="rules.txt"):
        """
        given the data find interesting rules
        """
        for level, item_list in self.freq_item_sets.iteritems():
            if level == 1:
                continue
            for itemset in item_list:
                rule = (itemset,[])
                self.generate_rules(rule, method)

        self.write_rules_to_file(filename, columns)

        #self.closed_freq_list = [k for k in self.closed_freq
        #                             if self.closed_freq[k] == 0]
        #self.maximal_freq_list = [k for k in self.maximal_freq
        #                              if self.maximal_freq[k] == 0]

        #print "CF:", self.closed_freq_list
        #print "MF:", self.maximal_freq_list
        return 


    
# add 1- itemsets to k-1 itemsets to genereate k- itemsets
class Apriori2(Apriori1):    
    def generate_candidates(self, data, level = 1):
        """
        generate candidate itemsets for a given level
        data is a numpy matrix, with binary values
        1 for occurence of an item 
        0 for non occurence of the item
        """
        
        size = data.shape[0]
        #print self.min_sup*size        
        if level == 1:
            # no itemsets have been genertated yet
            
            candidates = np.arange(data.shape[1])
            candidates = [[c] for c in candidates]
            self.candidates[level] = [candidate for candidate in candidates]
            #print level, self.candidates[level]
            return candidates
        
        else:
            prev_frequent = self.freq_item_sets[level-1]
            level_1_frequent = self.freq_item_sets[1]
            candidates = []
            for c_1 in prev_frequent:
                for c_2 in level_1_frequent:
                    can = c_1 + c_2
                    #[1,2] and [2,1] are same count once
                    # remove duplicates [1,2]+[1,3] => [1,2,3]
                    # according to algo
                    # sort both, if [:-2] same append last
                    # 2 generated
                    can = list(np.unique(np.sort(can)))
                    if can in candidates:
                        continue
                    #print c_1, c_2, can
                    candidates.append(can)

            self.candidates[level] = candidates
            return candidates

if __name__ == "__main__":

    if len(sys.argv) < 6:
        print "USAGE: python apriori.py <data_file> <min_sup> <min_conf> <methid> <implementation> <max_level>"
        print "\t data_file : [data/cars.csv, data/mushrooms.csv, data/nursery.scv]"
        print "\t min_sup: minuimum support threshold: 0-1 "
        print "\t min_conf: minimum confidence threshold: 0-1 "
        print "\t method: [lift , conf] "
        print "\t implementation: [prev, first] "
        print "\t max_level: [integer 0,1,2,3,...] "
        sys.exit()
        
    filename = sys.argv[1]
    min_sup  = float(sys.argv[2])
    min_conf = float(sys.argv[3])
    method = sys.argv[4]
    implement = sys.argv[5]
    max_level = int(sys.argv[6])
    
    if implement == "prev":
        apriori = Apriori1(min_sup= min_sup, min_conf = min_conf, max_level = max_level)
    else:
        apriori = Apriori2(min_sup= min_sup, min_conf = min_conf, max_level = max_level)

    df = pd.read_csv(filename)
    data = df.values
    columns = df.columns
    
    apriori.apriori_frequent_itemsets(data)

    total_candidate_count = sum([len(v)
                                 for k,v in apriori.candidates.iteritems()])
    total_frequent_count = sum([len(v)
                                for k,v in apriori.freq_item_sets.iteritems()])
    #total_cfi_count = len(apriori.closed_freq_list)
    #total_mfi_count = len(apriori.maximal_freq_list)
    total_cfi_count = len([k for k in apriori.closed_freq
                                     if apriori.closed_freq[k] == 0])
    total_mfi_count = len([k for k in apriori.maximal_freq
                                     if apriori.maximal_freq[k] == 0])
    print "\n\n"
    print "Total candidate item sets are:", total_candidate_count
    print "Total frequent item sets are:", total_frequent_count
    print "Total closed Frequent item sets are:", total_cfi_count
    print "Total maximal Frequnt item sets are:", total_mfi_count

    print "\nRULES are:\n" , 
    apriori.find_interesting_rules(method = method, columns = columns)
    print "TOTAL : ", len(apriori.interesting_rules)
