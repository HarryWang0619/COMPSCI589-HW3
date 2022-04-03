from sqlite3 import Row
from sklearn import datasets
import sklearn.model_selection
import random
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from collections import Counter

def importfile(name:str,delimit:str):
    # importfile('hw3_wine.csv', '\t')
    file = open("datasets/"+name, encoding='utf-8-sig')
    reader = csv.reader(file, delimiter=delimit)
    dataset = []
    for row in reader:
        dataset.append(row)
    file.close()
    return dataset

def same(attributecolumn):
    return all(item == attributecolumn[0] for item in attributecolumn)

def majority(attributecolumn):
    return np.argmax(np.bincount(attributecolumn.astype(int)))

def entropy(attributecol):
    values = list(Counter(attributecol).values())
    ent = 0
    for value in values:
        k = (value/sum(values))
        ent += -k*math.log(k,2)
    return ent

def gini(attributecol):
    values = list(Counter(attributecol).values())
    ginivalue = 1
    for value in values:
        prob = (value/sum(values))
        ginivalue -= prob**2
    return ginivalue

def dropbyindex(data, category, listindex):
    newdata = np.delete(data.T, listindex).T
    keytoremove = [list(category.keys())[i] for i in listindex]
    newcategory = category.copy()
    [newcategory.pop(key) for key in keytoremove]
    return newdata, newcategory

def id3bestseperate(dataset, attributes:dict):
    # dataset in is the dataset by row. 
    # attributes is the dictionary of attributes:type 
    # types: numerical, categorical, binary.
    datasetbycolumn = dataset.T
    classindex = list(attributes.values()).index("class")
    originalentrophy = entropy(datasetbycolumn[classindex])
    smallestentrophy = originalentrophy
    thresholdvalue = -1

    i = 0
    bestattribute = {list(attributes.keys())[i]:attributes[list(attributes.keys())[i]]}
    attributesinuse = list(attributes.keys())[1:] if (classindex == 0) else list(attributes.keys())[:classindex]
    # datasetinuse = datasetbycolumn[1:] if (classindex == 0) else datasetbycolumn[:classindex]

    for attribute in attributesinuse:
        idx = i+1 if classindex == 0 else i

        if attributes[attribute] == "categorical" or attributes[attribute] == "binary":
            listofkeys = list(Counter(datasetbycolumn[idx]).keys())
            listofcategory = [] # this is the list of categorical values.
            
            for key in listofkeys:
                indexlist = [idex for idex, element in enumerate(datasetbycolumn[idx]) if element == key]
                category = np.array(datasetbycolumn[classindex][indexlist])
                listofcategory.append(category)

            entropynow = 0

            for ctgry in listofcategory:
                a = len(ctgry)/len(datasetbycolumn[idx]) # This is probability
                entropynow += a * entropy(ctgry)

            if entropynow < smallestentrophy:
                smallestentrophy = entropynow
                bestattribute = {attribute:attributes[attribute]}
            
        elif attributes[attribute] == "numerical":
            datasetsort = datasetbycolumn.T[datasetbycolumn.T[:,idx].argsort(kind='quicksort')].T
            currentthreshold = (datasetsort[idx][1]+datasetsort[idx][0])/2
            k = 1
            while k < len(datasetsort.T):
                currentthreshold = (datasetsort[idx][k]+datasetsort[idx][k-1])/2
                listofcategory = [datasetsort[classindex][:k],datasetsort[classindex][k:]]
                entropynow = 0

                for ctgry in listofcategory:
                    a = len(ctgry)/len(datasetbycolumn[idx]) # This is probability
                    entropynow += a * entropy(ctgry)

                if entropynow < smallestentrophy:
                    smallestentrophy = entropynow
                    thresholdvalue = currentthreshold
                    bestattribute = {attribute:attributes[attribute]}    
                k += 1
        i += 1

    gain = originalentrophy-smallestentrophy
    # set first attribution dictionary {key:type} to the best attributes.
    return bestattribute, thresholdvalue, gain

def cartbestseperate(dataset, attributes:dict):
    # dataset in is the dataset by row. 
    # attributes is the dictionary of attributes:type 
    # types: numerical, categorical, binary.
    datasetbycolumn = dataset.T
    classindex = list(attributes.values()).index("class")
    originalgini = gini(datasetbycolumn[classindex])
    smallestgini = originalgini
    thresholdvalue = -1

    i = 0
    bestattribute = {list(attributes.keys())[i]:attributes[list(attributes.keys())[i]]}
    attributesinuse = list(attributes.keys())[1:] if (classindex == 0) else list(attributes.keys())[:classindex]
    # datasetinuse = datasetbycolumn[1:] if (classindex == 0) else datasetbycolumn[:classindex]

    for attribute in attributesinuse:
        idx = i+1 if classindex == 0 else i

        if attributes[attribute] == "categorical" or attributes[attribute] == "binary":
            listofkeys = list(Counter(datasetbycolumn[idx]).keys())
            listofcategory = [] # this is the list of categorical values.
            
            for key in listofkeys:
                indexlist = [idex for idex, element in enumerate(datasetbycolumn[idx]) if element == key]
                category = np.array(datasetbycolumn[classindex][indexlist])
                listofcategory.append(category)

            currentgini = 0

            for ctgry in listofcategory:
                a = len(ctgry)/len(datasetbycolumn[idx]) # This is probability
                currentgini += a * gini(ctgry)

            if currentgini < smallestgini:
                smallestgini = currentgini
                bestattribute = {attribute:attributes[attribute]}
            
        elif attributes[attribute] == "numerical":
            datasetsort = datasetbycolumn.T[datasetbycolumn.T[:,idx].argsort(kind='quicksort')].T
            currentthreshold = (datasetsort[idx][1]+datasetsort[idx][0])/2
            k = 1
            while k < len(datasetsort.T):
                currentthreshold = (datasetsort[idx][k]+datasetsort[idx][k-1])/2
                listofcategory = [datasetsort[classindex][:k],datasetsort[classindex][k:]]
                currentgini = 0

                for ctgry in listofcategory:
                    a = len(ctgry)/len(datasetbycolumn[idx]) # This is probability
                    currentgini += a * gini(ctgry)

                if currentgini < smallestgini:
                    smallestgini = currentgini
                    thresholdvalue = currentthreshold
                    bestattribute = {attribute:attributes[attribute]}    
                k += 1
        i += 1

    # set first attribution dictionary {key:type} to the best attributes.
    gain = originalgini-smallestgini
    return bestattribute, thresholdvalue, gain
