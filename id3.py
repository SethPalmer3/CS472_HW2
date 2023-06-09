#!/usr/bin/python3
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
import sys
import re
import math
# Node class for the decision tree
import node

train = None
varnames = None
test = None
testvarnames = None
root = None


# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
    if p <= 0 or p >= 1:
        return 0
    return -p*math.log2(p) - (1-p)*math.log2(1-p)

# Compute information gain for a particular split, given the counts
# py_pxi : number of positive hits in the attribute value
# pxi : number of occurances of the attribute value
# py : number of total positive hits in data set
# total : number of total of data set points(length of data)
def infogain(py_pxi, pxi, py, total):
    """Compute information gain for a particular split, given the counts
    Parameters
    ---
     py_pxi : number of positive hits in the attribute value
     pxi : number of occurances of the attribute value
     py : number of total positive hits in data set
     total : number of total of data set points(length of data)
     Returns
     ---
     int
        the info gain
    """
    # variable assignments for posititve hits
    positive_overall = py/total
    # entropy of positive hits
    ent_o = entropy(positive_overall)

    # in the case that there are no attribute values...
    if pxi == 0:
        p1 = (py - py_pxi) / (total)
        return ent_o -((total - pxi) / total * entropy(p1))

    # variable assignment for positve hits of attribute value
    positive_attr = py_pxi/pxi
    # entropy of attribute value, also conditional entropy
    ent_a = entropy(positive_attr)

    # attirbute value number matches total length of data
    if pxi == total:
        gain = ent_o-(pxi/ total)*ent_a
    else:
        p2 = abs((py - py_pxi) / (total - pxi))
        gain = ent_o -(pxi / total) *ent_a - ((total - pxi) / total * entropy(p2))
    return gain

def count_pos_hits(data, varname_id, attr_val = 1, pos_out_val = 1):
    """counts the number of of positive hits for a given attribute value with a desired output value. Assume last column in data is the output
    Parameters
    ---
    data : list[list]
        data matrix
    varname_id : int
        attribute id to check against
    attr_val : Any
        the value of the attribute to count positives against
    pos_out_val : Any
        which output value considered positive

    Returns
    ---
    (int, int)
        the number of positive hits in the attribute in the attribute value,
        and the total number of that attribute values.
        If the varname_id is equal to output id returns (0,0)"""
    hcount = 0
    vcount = 0
    if varname_id >= len(data[0])-1:
        return (0,0)
    for d in range(len(data)):
        if data[d][varname_id] == attr_val:
            vcount += 1 # value count
            if data[d][-1] == pos_out_val:
                hcount += 1 # hit count
    return (hcount, vcount)

def get_probabl_class(l):
    d = {}
    for x in l:
        if x not in d.keys():
           d[x] = 0
        d[x] += 1
    most_likely = None
    most_count = 0
    for attr_vals, count in d.items():
        if count > most_count:
            most_likely = attr_vals
            most_count = count
    return most_likely

def count_col(data, col_num):
    d = {
            1:0,
            0:0,
        }
    for r in range(len(data)):
        d[data[r][col_num]] += 1

    return d


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)

# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames: list[str]):
    # if the attribute name is None
    best_attr = -1
    best_infogain = 0
    total = len(data)
    set_dat = count_col(data, -1)
    for attr in range(len(varnames)-1): # Find the best info gain
        hit, tot = count_pos_hits(data, attr, 0, 0)
        temp_ig = infogain(hit, tot, set_dat[0], total)
        if temp_ig > best_infogain:
            best_attr = attr
            best_infogain = temp_ig
    # Base cases
    if best_infogain == 0:
        return node.Leaf(varnames, data[0][-1])
    
    data0 = []
    data1 = []
    #Split data
    for r in range(len(data)):
        if data[r][best_attr] == 0:
            data0.append(data[r])
        else:
            data1.append(data[r])

    left = build_tree(data0, varnames)
    right = build_tree(data1, varnames)

    return node.Split(varnames, best_attr, left, right)
 

# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS, testS, modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)


def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct) / len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print('Usage: python3 id3.py <train> <test> <model>')
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
