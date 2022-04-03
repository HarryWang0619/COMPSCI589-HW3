from utils import *
from decisiontree import *

# Stratified K-Fold method
def stratifiedkfold(data, categorydict, k = 10):
    classindex = list(categorydict.values()).index("class")
    datacopy = np.copy(data).T
    classes = list(Counter(datacopy[classindex]).keys())
    nclass = len(classes) # number of classes
    listofclasses = []

    for oneclass in classes:
        index = [idx for idx, element in enumerate(datacopy[classindex]) if element == oneclass]
        oneclassdata = np.array(datacopy.T[index])
        np.random.shuffle(oneclassdata)
        listofclasses.append(oneclassdata)

    splitted = [np.array_split(i, k) for i in listofclasses]
    nclass = len(classes)
    combined = []

    for j in range(k):
        ithterm = []
        for i in range(nclass):
            if ithterm == []:
                ithterm = splitted[i][j]
            else:
                ithterm = np.append(ithterm,splitted[i][j],0)
        combined.append(ithterm)
    
    return combined

# Bootstrap/Bagging method with resample ratio
def bootstrap(data, ratio=0.1): 
    data2 = np.copy(data)
    k = len(data)
    randomlist = random.sample(range(0, k), round(k*ratio))
    data2 = np.delete(data2, randomlist, 0)
    p = len(data2)
    randomfill = random.sample(range(0, p), k-p)
    data2 = np.concatenate((data2,data2[randomfill]),0)
    # print(len(data2))
    return data2

# Random Forest, plant a forest of n trees
def plantforest(data, categorydict, ntree=10, maxdepth=10, minimalsize=10, minimalgain=0.01, algortype='id3'):
    forest = []
    for i in range(ntree):
        datause = bootstrap(data, 0.05)
        tree = decisiontreeforest(datause,categorydict,algortype,maxdepth,minimalsize,minimalgain)
        forest.append(tree)
    return forest

# Predict the class of a single instance
def forestvote(forest, instance, categorydict):
    votes = {}
    for tree in forest:
        predict, correct, correctbool = prediction(tree,instance,categorydict)
        if predict not in votes:
            votes[predict] = 1
        else:
            votes[predict] += 1
    return max(votes, key=votes.get), correct

def importdata():
    cmc = importfile('cmc.data', ',')
    cmccategory = {"Wife's age":"numerical","Wife's education":"categorical",
    "Husband's education":"categorical","Number of children ever born":"numerical",
    "Wife's religion":"binary","Wife's now working?":"binary",
    "Husband's occupation":"categorical","Standard-of-living index":"categorical",
    "Media exposure":"binary","Contraceptive method used":"class"}
    cmcdata = np.array(cmc).astype(int)
    return cmcdata, cmccategory
    
cmcdata,cmccategory = importdata()

mytestforest = plantforest(cmcdata, cmccategory, ntree=70, maxdepth=5, minimalsize=10, minimalgain=0.001, algortype='gini')
tree01 = mytestforest[0]

cc = 0
for instance in cmcdata:
    predict,correct,bool = prediction(tree01,instance,cmccategory)
    # print(predict,correct,bool)
    if bool:
        cc += 1

print(cc/len(cmcdata))

cc2 = 0
for instance in cmcdata:
    predict, correct = forestvote(mytestforest,instance,cmccategory)
    #print(predict, correct)
    if predict == correct:
        cc2 += 1

print(cc2/len(cmcdata))