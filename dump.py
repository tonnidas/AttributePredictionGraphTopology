# This class is not runnable. It is mainly for dumping all previous codes just in case i cant find the right code on google

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



# ----------------------------------------------------------------------------------------------------------------------------------
# Read edges & attributes of testing graph from pickle (testing dataset (special case))
# with open('pickles/playGraph_edges.pickle', 'rb') as handle: data = pickle.load(handle)  
# with open('pickles/playGraph_attributes.pickle', 'rb') as handle: y = pickle.load(handle) 
# G = nx.from_pandas_edgelist(data, 'From', 'To')
# ----------------------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------------------
# read edges of UNC28 from pickle (UNC28 dataset (special case))
# with open('pickles/edges.pickle', 'rb') as handle: data = pickle.load(handle)  
# with open('pickles/attributes.pickle', 'rb') as handle: y = pickle.load(handle) 
# G = nx.from_pandas_edgelist(data, 'From', 'To')
# ----------------------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------------------
# Params:
#   G                 = networkx Graph
#   selected_features = parameter indicating the features to create
# Return values:
#   featuresDf = created feature dataframe
# Generate the features according to criteria of 'selected_features'
def get_features(G, selected_features):
    # Properties + adjacency
    if selected_features == '1':
        # nodePropertiesDf = getNodeProperties(G)
        with open('pickles/generated_nodeProperties/nodeProperties_UNC28.pickle', 'rb') as handle: nodePropertiesDf = pickle.load(handle)  
        adjDf = nx.to_pandas_adjacency(G)
        featuresDf = pd.concat([adjDf, nodePropertiesDf], axis=1)
        featuresDf.columns = featuresDf.columns.astype(str)
        return featuresDf

    # Features + adjacency

    # Properties (only)
    if selected_features == '7':
        # nodePropertiesDf = getNodeProperties(G)
        with open('pickles/generated_nodeProperties/nodeProperties_UNC28.pickle', 'rb') as handle: nodePropertiesDf = pickle.load(handle)  
        featuresDf = nodePropertiesDf
        return featuresDf
# ----------------------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------------------
# Params:
#   G                 = networkx Graph
#   selected_features = parameter indicating the features to create
# Return values:
#   featuresDf = created feature dataframe
#   y          = the attribute to be predicted
# def get_settings(dataset_attributes, dataset_edges, model, predicting_attribute, prediction_type, selected_features):
def get_settings(dataset, model, predicting_attribute, prediction_type, selected_features):
    # data, y = load_input_data(dataset_attributes, dataset_edges)
    # print("data: ===== ", data)
    # G = nx.from_pandas_edgelist(data, 'From', 'To')
    # y = y[predicting_attribute]   
    # featuresDf = get_features(G, selected_features)
    # print(y.shape)
    # print(featuresDf.shape)

    graph_file = 'Facebook100/fb100/' + dataset + '.graphml'
    G = nx.read_graphml(graph_file)                                       # Read the graph from 'Facebook100' folder
    featuresDf = get_features(G, selected_features)                       # Get generated node properties along with other required properties of the graph G
    yDF = pd.DataFrame.from_dict(G.nodes, orient='index')                 # Convert the dictionary of features to pandas dataframe
    y = yDF[predicting_attribute]                                         # get the attribute to be predicted

    return featuresDf, y
# ----------------------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------------------
G = nx.read_graphml('Facebook100/fb100/UNC28.graphml')
print(len(G.edges), len(G.nodes))
attribute_list = list(G.nodes(data=True))            # To see the features in dictionary mode
print(attribute_list[0])                             # To see the header of the features
f = pd.DataFrame.from_dict(G.nodes, orient='index')  # To convert features from dictionary type to pandas dataframe
print(f['gender'])                                   # To get a single column in a sorted manner
# ----------------------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------------------
print("Is G connected? = ", nx.is_connected(G), ". Then, How many components? = ", nx.number_connected_components(G))

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
# ----------------------------------------------------------------------------------------------------------------------------------