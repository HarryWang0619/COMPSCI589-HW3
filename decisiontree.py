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
    
    def isfather(self):
        if self.parent is None:
            return True
        else:
            return False
    

# Decision Tree that only analyze square root of the data.
def decisiontreeforest(dataset: np.array, dictattributes: dict, algortype: str ='id3', maxdepth: int = 10, minimalsize: int = 10, minimalgain: float = 0.01):
    datasetcopy = np.copy(dataset).T # dataset copy is by colomn. 
    dictattricopy = dictattributes.copy()
    classindex = list(dictattributes.values()).index("class")
    k = len(dictattributes)-1
    randomlist = random.sample(range(0, k), round(math.sqrt(k))) if classindex !=0 else random.sample(range(1, k+1), round(math.sqrt(k)))
    randomlist.append(classindex)
    randomkey = [list(dictattricopy.keys())[i] for i in randomlist]
    trimmeddict = {key:dictattricopy[key] for key in randomkey}
    trimmeddata = np.array(datasetcopy[randomlist])

    def processbest(algor):
        if algor == "cart" or algor == "gini":
            return cartbestseperate(trimmeddata.T, trimmeddict)
        else: # algor == "id3" or algor == "infogain"
            return id3bestseperate(trimmeddata.T, trimmeddict)

    node = Treenode(label=-1,type="decision")
    currentdepth = node.depth

    node.majority = majority(datasetcopy[classindex])

    if same(datasetcopy[classindex]):
        node.type = "leaf"
        node.label = datasetcopy[classindex][0]
        return node
    
    if len(dictattricopy) == 0:
        node.type = "leaf"
        node.label = majority(datasetcopy[classindex])
        return node

    # A stopping criteria  'minimal_size_for_split_criterion'

    if len(dataset) <= minimalsize:
        node.type = "leaf"
        node.label = majority(datasetcopy[classindex])
        return node

    bestattributedict,thresholdval,gain = processbest(algortype)
    bestattributename = list(bestattributedict.keys())[0]
    bestattributetype = bestattributedict[bestattributename]
    node.testattributedict = bestattributedict
    node.datatype = bestattributetype
    node.testattribute = bestattributename
    node.threshold = thresholdval
    bindex = list(dictattricopy.keys()).index(list(bestattributedict.keys())[0])

    # A Possible Stopping criteria 'minimal_gain'

    if gain < minimalgain:
        node.type = "leaf"
        node.label = majority(datasetcopy[classindex])
        return node

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
            subdata = np.delete(subdata,bindex,0)
            subdatalists.append(subdata.T)
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

        # Another Stoping criteria I could ADD: maximal depth
        
        if node.caldepth()+1 > maxdepth:  
            node.type = "leaf"
            node.label = node.majority
            node.threshold = thresholdval
            return node 

        subtree = decisiontreeforest(subvdata, dictattricopy, algortype, maxdepth, minimalsize, minimalgain)
        subtree.depth = currentdepth + 1
        subtree.parent = node
            
        if bestattributetype == 'numerical':
            attributevalue = "<=" if sdindex == 0 else ">"
        else:
            attributevalue = bigv[sdindex]

        edge[attributevalue] = subtree
        sdindex += 1

    node.edge = edge

    return node

# Predict the label of the test data, return correct and predict.
def prediction(tree: Treenode, instance, dictattricopy): # note that the instance is by row. (I formerly used by column)
    predict = tree.majority
    classindex = list(dictattricopy.values()).index("class")
    correct = instance[classindex]
    if tree.type == 'leaf':
        predict = tree.label
        return predict, correct, predict==correct

    testindex = list(dictattricopy.keys()).index(tree.testattribute)
    
    if tree.datatype == "numerical":
        if instance[testindex] <= tree.threshold:
            nexttree = tree.edge['<=']
        else:
            nexttree = tree.edge['>']
    else:
        if instance[testindex] not in tree.edge:
            return predict, correct, predict==correct
            
        nexttree = tree.edge[instance[testindex]]

    return prediction(nexttree, instance, dictattricopy)