# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
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

from main import do

res = do('UNC28', 'classification', 'RandomForest_hyper', 'student_fac', '2', 15)
fes = do('UNC28', 'classification', 'RandomForest_hyper', 'gender', '2', 15)
e_dict = dict()
for i in range(9):
    e_dict[i] = ['acc', round(res[i], 2), round(fes[i], 2)]

# resDf = pd.DataFrame(columns=['matric', 'Status', 'Gender'])
# resDf.loc[len(resDf.index)] = ['acc', round(res[0], 2), round(fes[0], 2)]
# resDf.loc[len(resDf.index)] = ['f1_macro', round(res[1], 2), round(fes[1], 2)]
# resDf.loc[len(resDf.index)] = ['precision_macro', round(res[2], 2), round(fes[2], 2)]
# resDf.loc[len(resDf.index)] = ['recall_macro', round(res[3], 2), round(fes[3], 2)]
# resDf.loc[len(resDf.index)] = ['f1_micro', round(res[4], 2), round(fes[4], 2)]
# resDf.loc[len(resDf.index)] = ['precision_micro', round(res[5], 2), round(fes[5], 2)]
# resDf.loc[len(resDf.index)] = ['recall_micro', round(res[6], 2), round(fes[6], 2)]
# resDf.loc[len(resDf.index)] = ['f1_weighted', round(res[7], 2), round(fes[7], 2)]
# resDf.loc[len(resDf.index)] = ['adj_ri', round(res[8], 2), round(fes[8], 2)]

resDf = pd.DataFrame.from_dict(e_dict, orient='index')
# resDf = resDf.set_index('matric')
resDf.to_excel('result.xlsx') 
print(resDf)