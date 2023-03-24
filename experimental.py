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

from config import do_category_specific_task_prediction

data = ['American75', 'Amherst41', 'Auburn71', 'Baylor93', 'Berkeley13', 'Bingham82', 'Bowdoin47', 'Brandeis99', 'Brown11', 'BU10']
for data_name in data:
    graph_file = 'Facebook100/fb100/' + data_name + '.graphml'
    nxGraph = nx.read_graphml(graph_file)  # Read the graph from 'Facebook100' folder    
    print(data_name, " = ", len(list(nxGraph.nodes)), len(list(nxGraph.edges())))