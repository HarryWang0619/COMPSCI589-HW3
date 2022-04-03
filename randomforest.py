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
            if len(ithterm) == 0:
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
def plantforest(data, categorydict, ntree=10, maxdepth=10, minimalsize=10, minimalgain=0.01, algortype='id3', bootstrapratio = 0.1):
    forest = []
    for i in range(ntree):
        datause = bootstrap(data, bootstrapratio)
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

# A complete k-fold cross validation
def kfoldcrossvalid(data, categorydict, k=10, ntree=10, maxdepth=5, minimalsize=10, minimalgain=0.01, algortype='id3', bootstrapratio = 0.1):
    folded = stratifiedkfold(data, categorydict, k)
    listofnd = []
    accuracylist = []
    for i in range(k):
        print("at fold", i)
        testdataset = folded[i]
        foldedcopy = folded.copy()
        foldedcopy.pop(i)
        traindataset = np.vstack(foldedcopy) 
        correctcount = 0
        trainforest = plantforest(traindataset,categorydict,ntree,maxdepth,minimalsize,minimalgain,algortype,bootstrapratio)
        emptyanalysis = []
        # testdataset = traindataset
        for instance in testdataset:
            predict, correct = forestvote(trainforest,instance,categorydict)
            emptyanalysis.append([predict, correct])
            if predict == correct:
                correctcount += 1
        listofnd.append(np.array(emptyanalysis))
        print('fold', i, ' accuracy: ', correctcount/len(testdataset))
        accuracylist.append(correctcount/len(testdataset))
    acc = np.mean(accuracylist)
    return acc


def importhousedata():
    house = importfile('hw3_house_votes_84.csv', ',')
    housecategory = {}
    for i in house[0]:
        housecategory[i] = 'categorical'
    housecategory["class"] = 'class'
    housedata = np.array(house[1:]).astype(float)
    return housedata, housecategory

def importwinedata():
    wine = importfile('hw3_wine.csv', '\t')
    winecategory = {}
    for i in wine[0]:
        winecategory[i] = 'numerical'
    winecategory["# class"] = 'class'
    winedata = np.array(wine[1:]).astype(float)
    return winedata, winecategory

def importcancerdata():
    cancer = importfile('hw3_cancer.csv', '\t')
    cancercategory = {}
    for i in cancer[0]:
        cancercategory[i] = 'numerical'
    cancercategory["Class"] = 'class'
    cancerdata = np.array(cancer[1:]).astype(float)
    return cancerdata, cancercategory

def importcmcdata():
    cmc = importfile('cmc.data', ',')
    cmccategory = {"Wife's age":"numerical","Wife's education":"categorical",
    "Husband's education":"categorical","Number of children ever born":"numerical",
    "Wife's religion":"binary","Wife's now working?":"binary",
    "Husband's occupation":"categorical","Standard-of-living index":"categorical",
    "Media exposure":"binary","Contraceptive method used":"class"}
    cmcdata = np.array(cmc).astype(int)
    return cmcdata, cmccategory
    
housedata, housecategory = importhousedata()
winedata, winecategory = importwinedata()
cancerdata, cancercategory = importcancerdata()
cmcdata,cmccategory = importcmcdata()

xval = kfoldcrossvalid(cancerdata, cancercategory, k=10, ntree=20, maxdepth=10, minimalsize=10, minimalgain=0.01, algortype='id3', bootstrapratio = 0.1)
print(xval)
# mytestforest = plantforest(cmcdata, cmccategory, ntree=150, maxdepth=7, minimalsize=10, minimalgain=0.001, algortype='id3', bootstrapratio=0.15)
# tree01 = mytestforest[0]

# cc = 0
# for instance in cmcdata:
#     predict,correct,bool = prediction(tree01,instance,cmccategory)
#     # print(predict,correct,bool)
#     if bool:
#         cc += 1

# print(cc/len(cmcdata))

# cc2 = 0
# for instance in cmcdata:
#     predict, correct = forestvote(mytestforest,instance,cmccategory)
#     #print(predict, correct)
#     if predict == correct:
#         cc2 += 1

# print(cc2/len(cmcdata))