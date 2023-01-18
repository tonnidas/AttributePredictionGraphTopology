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


# Read edges & attributes of testing graph from pickle (testing dataset (special case))
# with open('pickles/playGraph_edges.pickle', 'rb') as handle: data = pickle.load(handle)  
# with open('pickles/playGraph_attributes.pickle', 'rb') as handle: y = pickle.load(handle) 
# G = nx.from_pandas_edgelist(data, 'From', 'To')

# read edges of UNC28 from pickle (UNC28 dataset (special case))
# with open('pickles/edges.pickle', 'rb') as handle: data = pickle.load(handle)  
# with open('pickles/attributes.pickle', 'rb') as handle: y = pickle.load(handle) 
# G = nx.from_pandas_edgelist(data, 'From', 'To')

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