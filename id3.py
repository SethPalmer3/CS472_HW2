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
        var = {} # stores collection info for a particular attribute
        for c in range(len(data)):
            if var.get(data[c][v]) is None: # Not already stored value
                var[data[c][v]] = 0
            var[data[c][v]] += 1
        var_counts[varnames[v]] = var

    return var_counts

def count_pos_hits(data, varname_id, attr_val, pos_out_val):
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
    if varname_id == len(data[0])-1:
        return (0,0)
    for d in range(len(data)):
        if data[d][varname_id] == attr_val:
            vcount += 1 # value count
            if data[d][-1] == pos_out_val:
                hcount += 1 # hit count
    return (hcount, vcount)

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
    inc_infogain = -1
    best_var = ""
    stats = collect_counts(data, varnames)
    out_val = list(list(stats.items())[-1][1].keys())[0] # Gets the outputs positive value
    out_pos_hits = list(list(stats.items())[-1][1].values())[0] # Gets the number of the outputs positive value
    for attr, cnts in stats.items(): # Find highest info gain
        val_pos = list(cnts.items())[-1][0] # Get the attribute value
        hits, tot = count_pos_hits(data, varnames.index(attr), val_pos, out_val)
        infgn = infogain(hits, tot, out_pos_hits, len(data))
        if infgn > inc_infogain:
            best_var = attr
            inc_infogain = infgn
    return best_var

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
    v1 = []
    v2 = []
    id1 = [x for x in range(varname_index)] # First half
    id2 = [x for x in range(varname_index + 1, len(data[0]))] # Second half
    vals = collect_vals(data, varname_index)
    for row in range(len(data)):
        if data[row][varname_index] == vals[0]:
            v1.append(data[row][:varname_index])
        else:
            v2.append(data[row][varname_index+1:])
    return ((id1, v1), (id2, v2))

def get_col(data, col_num):
    """Gets all the column data in the col_num th column
    Parameters
    ---
    data : list[list]
        data to pull a column from
    col_num : int
        the column number of which to grab

    Returns
    ---
    list
        a list of data from that column(in order from highest row to lowest)
    """
    ret = []
    for r in range(len(data)):
        ret.append(data[r][col_num])

    return ret

def same_class(outclass):
    """Determines if all outputs belong to the same class
    Parameters
    ---
    outclass : list
        a list of the output

    Returns
    ---
    bool
        true if all outputs are the same class
    """
    for o in range(1,len(outclass)):
        if outclass[o] == outclass[o-1]:
            return False

    return True

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
    # return node.Leaf(varnames, 1)

    py_pxi = 0        # number of postive hits in attribute value
    pxi = 0           # number of occurances of the attribute value
    py = 0            # number of total positve hits in data set
    total = len(data) # total of data set (length of data)
    gain = 0          # current value of info gain
    gain_name = None  # name of info gain attribute

    for i in range in range(len(varnames) -1):
        for j in data:
            # check attribute 
            if j[i] == 1:
                pxi += 1
            # check data set
            if j[-1] == 1:
                py += 1
            # data set and attribute 
            if j[i] == 1 and j[-1] == 1:
                py_pxi += 1

    # base cases for total postive hits
    if py == total:
        return node.Leaf(varnames, 1)
    if py == 0:
        return node.Leaf(varnames, 0)

    # get current info gain value
    for d in range(len(varnames) - 1):
        temp_g = infogain(py_pxi, pxi, py, total)
        if temp_g > gain:
            gain = temp_g
            gain_name = d
    
    # if the attribute name is None
    best_split = gain_name
    if best_split is None:
        return node.Leaf(varnames, 1)
    
    # divide the data
    data0 = []
    data1 = []

    for i in range(len(data)):
        if data[i][gain_name] == 0:
            list = data[i]
            data0.append(list)
        else:
            list = data[i]
            data1.append(list)

    return node.Split(varnames, gain_name, build_tree(data0, varnames), build_tree(data1, varnames))
 

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
