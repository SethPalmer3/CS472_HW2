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
    return -p*math.log2(p) - (1-p)*math.log2(1-p)

# Compute information gain for a particular split, given the counts
# py_pxi : number of positive hits in the attribute value x_i
# pxi : number of attribute value x_i
# py : number of positive hits
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    # variable assignments for posititve hits
    positive_overall = py/total
    # entropy of positive hits
    ent_o = entropy(positive_overall)

    # in the case that there are no attribute values...
    if pxi == 0:
        p1 = (py - py_pxi) / (total)
        return ent_o -((total - pxi) / total * entropy(p1))
        p1 = (py_pxi - py) / (total)
        return ent_o -((total) / total * entropy(p1))

    # variable assignment for positve hits of attribute value
    positive_attr = py_pxi/pxi
    positive_attr = pxi/py_pxi
    # entropy of attribute value, also conditional entropy
    ent_a = entropy(positive_attr)

    # attirbute value number matches total length of data
    if pxi == total:
        gain = ent_o-(pxi/ total)*ent_a
    else:
        p2 = (py - py_pxi) / (total - pxi)
        p2 = (py_pxi - py) / (total - pxi)
        gain = ent_o -(pxi / total) *ent_a - ((total - pxi) / total * entropy(p2))
    return gain

def collect_vals(data, varname, varnames=[]):
    """Gather a list of unique values for a given attribute name or id
    Parameters
    ---
    data : list
        data with values of varname
    varname : int | str
        the identifier of the desired attribute
    [varnames] : list
        the list of varnames, only needed if varname is string

    Returns
    ---
    list
        the collection of unique values
    """

    vals = []
    if type(varname) == str:
        id = varnames.index(varname)
    else:
        id = varname

    for c in range(len(data)):
        if data[c][id] not in vals:
            vals.append(data[c][id])
    return vals

# OTHER SUGGESTED HELPER FUNCTIONS:
def collect_counts(data: list[list], varnames: list[str])->dict[str, dict]:
    """Creates a dictionary of dictionaries for which each entry describes the attributes values and how many there are

    Parameters
    ----------
    data : list
        A 2D list of data values where the columns consist of values for a given attribute
    varnames : list[str]
        A list of all the attribute names that are associated with the data

    Returns
    -------
    dict
        where each entry describes the attribute and the value is another dict with it's values and how many there are
    """
    var_counts = {}
    for v in range(len(varnames)):
        var = {}
        for c in range(len(data)):
            if var.get(data[c][v]) is None:
                var[data[c][v]] = 0
            var[data[c][v]] += 1
        var_counts[varnames[v]] = var

    return var_counts

# - find the best variable to split on, according to mutual information
def best_split_attr(data, varnames):
    """ Determines the best attribute to split the given data on
    Parameters
    ---
    data : list
        The data set for each attribute
    varnames : list[str]
        The list of attribute names 

    Returns
    ---
    str
        the string in the varname on which to do a split on
    """
    pass

# - partition data based on a given variable
def partition_on_attr(data, varname_index):
    """ Split the data based on the given attribute
    Parameters
    ---
    data: list
        data to split on
    varname: int
        split databased on attribute id

    Returns
    ---
    ((list[int], list), (list[int], list))
        returns a tuple of tuples each of which the first element is the list of attribute id,
        and the second element is the list of data
    """
    v1 = [], v2 = []
    id1 = [x for x in range(varname_index)] # First half
    id2 = [x for x in range(varname_index + 1, len(data[0]))] # Second half
    vals = collect_vals(data, varname_index)
    for row in range(len(data)):
        if data[row][varname_index] == vals[0]:
            v1.append(data[row][:varname_index] + data[row][varname_index+1:])
        else:
            v2.append(data[row][:varname_index] + data[row][varname_index+1:])
    return ((id1, v1), (id2, v2))

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
def build_tree(data, varnames):
    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return a leaf predicting "1":
    return node.Leaf(varnames, 1)


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
