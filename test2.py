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


# read edges from pickle (large dataset)
with open('pickles/UNC28_edges.pickle', 'rb') as handle: data = pickle.load(handle)  
with open('pickles/UNC28_attributes.pickle', 'rb') as handle: y = pickle.load(handle) 


# print("Status = ", y['Status'].unique(), "How many classes? = ", len(y['Status'].unique()) )
# print("Gender = ", y['Gender'].unique(), "How many classes? = ", len(y['Gender'].unique()) )
# print("Major = ", y['Major'].unique(), "How many classes? = ", len(y['Major'].unique()) )
# print("Minor = ", y['Minor'].unique(), "How many classes? = ", len(y['Minor'].unique()) )
# print("Dorm = ", y['Dorm'].unique(), "How many classes? = ", len(y['Dorm'].unique()) )
# print("Graduation_Year = ", y['Graduation_Year'].unique(), "How many classes? = ", len(y['Graduation_Year'].unique()) )
# print("High_school = ", y['High_school'].unique(), "How many classes? = ", len(y['High_school'].unique()) )


G = nx.read_graphml('Facebook100/fb100/UNC28.graphml')
print(len(G.edges), len(G.nodes))
attribute_list = list(G.nodes(data=True))
print(attribute_list[0])

f = pd.DataFrame.from_dict(G.nodes, orient='index')
print(f['gender'])


print("Is G connected? = ", nx.is_connected(G), ". Then, How many components? = ", nx.number_connected_components(G))

adj = G.edges
adj = nx.adjacency_matrix(G)
y = nx.get_node_attributes(G, 'gender')
# print("y of Facebook100/fb100/UNC28.graphml = ", y)
# print(nx.attr_matrix(G))

# print(y.values())

# from constructor import load_input_data
# data, y = load_input_data('UNC28_attributes', 'UNC28_edges')
# # print("data: ===== ", data)
# G = nx.from_pandas_edgelist(data, 'From', 'To')

# y = y['Gender'] 
# print('y = ', y)

# # Store adj in a pickle 
# with open('adj_UNC28.pickle', 'wb') as handle: pickle.dump(adj, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print("adj_UNC28 = ", adj)

# # read edges from pickle (large dataset)
# with open('pickles/UNC28_edges.pickle', 'rb') as handle: data = pickle.load(handle)  

# print('data = ', data)




# # read edges from pickle (large dataset)
# with open('adj_American75.pickle', 'rb') as handle: data = pickle.load(handle)  
