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