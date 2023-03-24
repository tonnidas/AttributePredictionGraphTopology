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



# Temporarily commented. Remove comment when 'generated_nodeProperties' collection done
# --------------------------------------------
# G_name = 'timepoint0_baseline_graph'
# file_pickle = '{}.pickle'.format(G_name)
# with open(file_pickle, 'rb') as handle: G = pickle.load(handle) 

# edgelist = [(1,2), (3,4), (2,4), (4,5), (3,5), (6,7)]
# G = nx.from_edgelist(edgelist)
# --------------------------------------------