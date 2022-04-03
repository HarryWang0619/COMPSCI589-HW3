import sklearn.model_selection
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import random
from collections import Counter
from utils import *

class Treenode:
    type = ""
    datatype = ""
    label = None
    testattribute = ""
    edge = {}
    majority = -1
    threshold = -1 # for numerical value
    testattributedict = {}
    depth = 0
    _caldepth = 0
    
    parent = None

    def __init__(self, label, type):
        self.label = label
        self.type = type
        # self.left = left
        # self.right = right

    def caldepth(self):
        a = self
        while a.parent is not None:
            self._caldepth += 1
            a = a.parent
        return self._caldepth

# Decision Tree
def decisiontree(dataset: np.array, dictattributes: dict, algortype: str ='id3'):
    datasetcopy = np.copy(dataset).T # dataset copy is by colomn. 
    dictattricopy = dictattributes.copy()
    classindex = list(dictattricopy.values()).index("class")

    def processbest(algor):
        if algor == "cart" or algor == "gini":
            return cartbestseperate(datasetcopy.T, dictattricopy)
        else: # algor == "id3" or algor == "infogain"
            return id3bestseperate(datasetcopy.T, dictattricopy)

    node = Treenode(label=-1,type="decision")

    node.majority = majority(datasetcopy[classindex])

    if same(datasetcopy[classindex]):
        node.type = "leaf"
        node.label = datasetcopy[classindex][0]
        return node
    
    if len(dictattricopy) == 0:
        node.type = "leaf"
        node.label = majority(datasetcopy[classindex])
        return node

    bestattributedict,thresholdval = processbest(algortype)[:2]
    bestattributename = list(bestattributedict.keys())[0]
    bestattributetype = bestattributedict[bestattributename]
    node.testattributedict = bestattributedict
    node.datatype = bestattributetype
    node.testattribute = bestattributename
    node.threshold = thresholdval
    bindex = list(dictattricopy.keys()).index(list(bestattributedict.keys())[0])

    subdatalists = []
    if bestattributetype == "numerical":
        sortedcopy = datasetcopy.T[datasetcopy.T[:,bindex].argsort(kind='quicksort')].T
        splitindex = 0
        for numericalvalue in sortedcopy[bindex]:
            if numericalvalue > thresholdval:
                break
            else:
                splitindex += 1
        subdatalistraw = [sortedcopy.T[:splitindex].T,sortedcopy.T[splitindex:].T]
        for subdata in subdatalistraw:
            subdatav = np.delete(subdata,bindex,0)
            subdatalists.append(subdatav.T)
    else:
        bigv = list(Counter(datasetcopy[bindex]).keys()) # this is the all the categories of the test attribute left.
    
        for smallv in bigv:
            index = [idx for idx, element in enumerate(datasetcopy[bindex]) if element == smallv]
            subdatav = np.array(datasetcopy.T[index]).T
            subdatav = np.delete(subdatav,bindex,0)  # I delete the column I already used using bindex as reference. 
            # Then, later, pop the same index from list attribute.
            subdatalists.append(subdatav.T) # list of nparrays of target/label/categories.

    dictattricopy.pop(bestattributename)
    
    edge = {}
    sdindex = 0
    for subvdata in subdatalists:
        if subvdata.size == 0:
            node.type = "leaf"
            node.label = node.majority
            node.threshold = thresholdval
            return node

        subtree = decisiontree(subvdata, dictattricopy, algortype)
        if bestattributetype == 'numerical':
            attributevalue = "<=" if sdindex == 0 else ">"
        else:
            attributevalue = bigv[sdindex]

        edge[attributevalue] = subtree
        sdindex += 1

    node.edge = edge

    return node

# Tell me whether a single result is correct or not.
def prediction(tree: Treenode, instance, dictattricopy): # note that the instance if by row. (I formerly used by column)
    predict = tree.majority
    classindex = list(dictattricopy.values()).index("class")
    correct = instance[classindex]
    if tree.type == 'leaf':
        predict = tree.label
        return predict==correct, predict, correct

    testindex = list(dictattricopy.keys()).index(tree.testattribute)
    
    if tree.datatype == "numerical":
        if instance[testindex] <= tree.threshold:
            nexttree = tree.edge['<=']
        else:
            nexttree = tree.edge['>']
    else:
        if instance[testindex] not in tree.edge:
            return predict==correct, predict, correct
            
        nexttree = tree.edge[instance[testindex]]

    return prediction(nexttree, instance, dictattricopy)
   