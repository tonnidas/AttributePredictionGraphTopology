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
from numpy import linalg as LA

# ----------------------------------------
def eigen(G):
    adj = nx.adjacency_matrix(G).todense()

    matrix1 = adj
    matrix2 = np.ones(adj.shape[0])

    tol = 0.000001
    max_iter = 100
    prevNormalizedValue = -1000

    for i in range(max_iter):

        res = np.matmul(matrix1, matrix2)
        normalizedValue = LA.norm(res)

        if abs(prevNormalizedValue - normalizedValue) < tol:
            break
        
        res = res / normalizedValue

        matrix2 = res
        prevNormalizedValue = normalizedValue

    print('Iterations:', i)
    
    return matrix2

# --------------------------------------------

# ----------------------------------------
def eigen2(G):
    adj = nx.adjacency_matrix(G).todense()
    print('Adj shape:', adj.shape)

    matrix1 = adj
    matrix2 = np.ones(adj.shape[0])

    tol = 0.000001
    max_iter = 100
    prevNormalizedValue = -1000

    for i in range(max_iter):

        res = np.matmul(matrix1, matrix2)
        normalizedValue = LA.norm(res)
        res = res / normalizedValue

        err = sum(abs(res - matrix2))
        if err < adj.shape[0] * tol:
            return res

        matrix2 = res        

    print('Iterations:', i)
    
    return matrix2

# --------------------------------------------


# --------------------------------------------
# G_name = 'timepoint0_baseline_graph'
# file_pickle = '{}.pickle'.format(G_name)
# with open(file_pickle, 'rb') as handle: G = pickle.load(handle) 

edgelist = [(1,2), (3,4), (2,4), (4,5), (3,5)]
G = nx.from_edgelist(edgelist)
# --------------------------------------------


# Generate eigen vector using nx
eigen_nx = pd.DataFrame.from_dict(nx.eigenvector_centrality(G), orient='index', columns=['eigenvector_centrality'])
print(eigen_nx)

# Generate eigen vector using custom method
eigen_custom = eigen(G)
print(eigen_custom)

# Generate eigen vector using custom method 2
eigen_custom_2 = eigen2(G)
print(eigen_custom_2)
