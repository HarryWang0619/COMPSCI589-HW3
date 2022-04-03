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
    print(len(data2))
    return data2


