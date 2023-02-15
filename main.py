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

from config import do_category_specific_task_prediction

data_name = 'Amherst41'

for j in range(1, 10):
    res = do_category_specific_task_prediction(data_name, 'classification', 'RandomForest_hyper', 'student_fac', str(j), 15)
    fes = do_category_specific_task_prediction(data_name, 'classification', 'RandomForest_hyper', 'gender', str(j), 15)

    e_dict = dict()
    matric_name = ['acc', 'f1_macro', 'precision_macro', 'recall_macro', 'f1_micro', 'precision_micro', 'recall_micro', 'f1_weighted', 'adj_RI']
    for i in range(9):
        e_dict[i] = [matric_name[i], round(res[i], 6), round(fes[i], 6)]

    resDf = pd.DataFrame.from_dict(e_dict, orient='index')
    f = 'Result/' + data_name + '/' + data_name + '_category_' + str(j) + '.xlsx'
    resDf.to_excel(f) 
    print(resDf)