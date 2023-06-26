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

# data = ['American75', 'Amherst41', 'Auburn71', 'Baylor93', 'Berkeley13', 'Bingham82', 'Bowdoin47', 'Brandeis99', 'Brown11', 'BU10']
# 'student_fac' or 'gender' or 'major_index' or 'second_major' or 'dorm' or 'year' or 'high_school  ('Status' = 'student_fac')

data = ['Bowdoin47']


# for data_name in data:
#     graph_file = '../graph-data/Facebook100/Raw/' + data[0] + '.graphml'
#     nxGraph = nx.read_graphml(graph_file)  # Read the graph from 'Facebook100' folder    
#     # print(data_name, " = ", len(list(nxGraph.nodes)), len(list(nxGraph.edges())))

#     features = nx.get_node_attributes(nxGraph, "high_school")
#     # print(len(color), len(nxGraph.nodes()))
#     uniques = set(features.values())
#     print(len(uniques))

with open('pickles/generated_nodeProperties/nodeProperties_{}.pickle'.format(data[0]), 'rb') as handle: node = pickle.load(handle)
print(node)