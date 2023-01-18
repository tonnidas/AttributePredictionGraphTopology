# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
np.seterr(divide='ignore', invalid='ignore')    # for allowing divide by zero
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import csr_matrix
import pickle
import os

from sknetwork.path.shortest_path import get_distances
from typing import Union, Optional
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from helperMethods import *
# ----------------------------------------



# -------------------------
### Prepare small dataset
with open('pickles/UNC28_edges.pickle', 'rb') as handle: data = pickle.load(handle)
print(data)
data = data[(data['From'] < 100) & (data['To'] < 100)]

for i in range(100):
    data = data.append({'From': i, 'To': i}, ignore_index = True)
print(data)
with open('pickles/playGraph_edges.pickle', 'wb') as handle: pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('pickles/UNC28_attributes.pickle', 'rb') as handle: data = pickle.load(handle)
print(data)
data = data[0:100]
print(data)
with open('pickles/playGraph_attributes.pickle', 'wb') as handle: pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('pickles/playGraph_edges.pickle', 'rb') as handle: data = pickle.load(handle)
print(data)

with open('pickles/playGraph_attributes.pickle', 'rb') as handle: data = pickle.load(handle)
print(data)
# -------------------------