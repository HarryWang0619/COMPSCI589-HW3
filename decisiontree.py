import sklearn.model_selection
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from collections import Counter
from utils import *

class Treenode:
    type = ""
    label = None
    testattribute = ""
    edge = {}
    majority = -1
    threshold = -1 # for numerical value

    def __init__(self, label, type):
        self.label = label
        self.type = type
        # self.left = left
        # self.right = right


# Decision Tree
def decisiontree(dataset: np.array, listattributes: list, algortype: str ='id3'):
    def processbest(algor):
        if algor == "id3" or algor == "infogain":
            return id3(datasetcopy, listattricopy)
        elif algor == "cart" or algor == "gini":
            return cart(datasetcopy, listattricopy)
        else: 
            return cart(datasetcopy, listattricopy)

    datasetcopy = np.copy(dataset)
    listattricopy = listattributes.copy()
    
    node = Treenode(label=-1,type="decision")

    node.majority = majority(datasetcopy[-1])

    if same(datasetcopy[-1]):
        node.type = "leaf"
        node.label = datasetcopy[-1][0]
        return node
    
    if len(listattricopy) == 0:
        node.type = "leaf"
        node.label = majority(datasetcopy[-1])
        return node

    bestattribute = processbest(algortype)[0]
    node.testattribute = bestattribute
    bindex = listattricopy.index(bestattribute)

    bigv = list(Counter(datasetcopy[bindex]).keys())

    subdatalists = []
    for smallv in bigv:
        index = [idx for idx, element in enumerate(datasetcopy[bindex]) if element == smallv]
        subdatav = np.array(datasetcopy.T[index]).T
        subdatav = np.delete(subdatav,bindex,0)  # I delete the column I already used using bindex as reference. 
        # Then, later, pop the same index from list attribute.
        subdatalists.append(subdatav) # list of nparrays of target/label/categories.

    listattricopy.pop(bindex)
    
    edge = {}
    sdindex = 0
    for subvdata in subdatalists:
        if subvdata.size == 0:
            node.type = "leaf"
            node.label = majority(subdatav[-1])

        subtree = decisiontree(subvdata, listattricopy, algortype)
        attributevalue = bigv[sdindex]
        edge[attributevalue] = subtree
        sdindex += 1

    node.edge = edge

    return node