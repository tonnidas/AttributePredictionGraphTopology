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


# ----------------------------------------
def getNodeProperties(G):
    print('calculating degree')
    temp_dict = dict(G.degree)
    temp_max = max(temp_dict.values())
    degree_regularized = {key: value / temp_max for key, value in temp_dict.items()}
    df1 = pd.DataFrame.from_dict(degree_regularized, orient='index', columns=['degree'])

    print('calculating degree_centrality')
    df2 = pd.DataFrame.from_dict(nx.degree_centrality(G), orient='index', columns=['degree_centrality'])

    print('calculating clustering_coefficient')
    df3 = pd.DataFrame.from_dict(nx.clustering(G), orient='index', columns=['clustering_coefficient'])

    print('calculating eccentricity')
    print("Is G connected? = ", nx.is_connected(G), ". Then, How many components? = ", nx.number_connected_components(G))
    graphs = list(G.subgraph(c) for c in nx.connected_components(G)) # returns a list of disconnected graphs as subgraphs
    dict_1 = {}
    for subgraph in graphs:
        dict_2 = nx.eccentricity(subgraph)
        dict_1 = {**dict_1,**dict_2}
    max_ecc = max(dict_1.values())
    ecc_regularized = {key: value / max_ecc for key, value in dict_1.items()}
    df4 = pd.DataFrame.from_dict(ecc_regularized, orient='index', columns=['eccentricity'])
    print(df4)

    print('calculating closeness_centrality')
    df5 = pd.DataFrame.from_dict(nx.closeness_centrality(G), orient='index', columns=['closeness_centrality'])

    print('calculating betweenness_centrality')
    df6 = pd.DataFrame.from_dict(nx.betweenness_centrality(G), orient='index', columns=['betweenness_centrality'])

    print('done calculating node properties')

    return pd.concat([df1, df2, df3, df4, df5, df6], axis=1)
# ----------------------------------------

# read edges from pickle (small dataset)
# with open('pickles/playGraph_edges.pickle', 'rb') as handle: data = pickle.load(handle)  
# with open('pickles/playGraph_attributes.pickle', 'rb') as handle: y = pickle.load(handle) 

# read edges from pickle (large dataset)
with open('pickles/edges.pickle', 'rb') as handle: data = pickle.load(handle)  
with open('pickles/attributes.pickle', 'rb') as handle: y = pickle.load(handle) 

# generate features
G = nx.from_pandas_edgelist(data, 'From', 'To')
nodePropertiesDf = getNodeProperties(G)
print(nodePropertiesDf)

# store generated features of small dataset
# with open('pickles/playGraph_nodeProperties.pickle', 'wb') as handle: pickle.dump(featuresDf, handle, protocol=pickle.HIGHEST_PROTOCOL)

# store generated features of large dataset
with open('pickles/nodeProperties.pickle', 'wb') as handle: pickle.dump(featuresDf, handle, protocol=pickle.HIGHEST_PROTOCOL)

# adjDf = nx.to_pandas_adjacency(G)
# featuresDf = pd.concat([adjDf, nodePropertiesDf], axis=1)