# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
np.seterr(divide='ignore', invalid='ignore')    # for allowing divide by zero
import matplotlib.pyplot as plt
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
    # print('calculating degree')
    # df1 = pd.DataFrame.from_dict(dict(G.degree), orient='index', columns=['degree'])

    # print('calculating degree_centrality')
    # df2 = pd.DataFrame.from_dict(nx.degree_centrality(G), orient='index', columns=['degree_centrality'])

    # print('calculating clustering_coefficient')
    # df3 = pd.DataFrame.from_dict(nx.clustering(G), orient='index', columns=['clustering_coefficient'])

    # print('calculating eccentricity')
    # df4 = pd.DataFrame.from_dict(nx.eccentricity(G), orient='index', columns=['eccentricity'])

    print('calculating closeness_centrality')
    df5 = pd.DataFrame.from_dict(nx.closeness_centrality(G), orient='index', columns=['closeness_centrality'])

    # print('calculating betweenness_centrality')
    # df6 = pd.DataFrame.from_dict(nx.betweenness_centrality(G), orient='index', columns=['betweenness_centrality'])

    print('done calculating node properties')

    # return pd.concat([df1, df2, df3, df4, df5, df6], axis=1)
    return pd.concat([df5], axis=1)

# with open('pickles/edges.pickle', 'rb') as handle: data = pickle.load(handle)   # read edges from pickle (full dataset)
# G = nx.from_pandas_edgelist(data, 'From', 'To')

# G = nx.Graph([(1, 2), (1, 3), (2, 4), (2, 5), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (5, 7), (6, 8), (9, 10), (9, 12), (10, 12), (11, 12)])

# with open('pickles/playGraph_edges.pickle', 'rb') as handle: data = pickle.load(handle)   # read edges from pickle (small dataset)
# G = nx.from_edgelist(data)

# nodePropertiesDf = getNodeProperties(G)
# adjDf = nx.to_pandas_adjacency(G)
# featuresDf = pd.concat([adjDf, nodePropertiesDf], axis=1)

# with open('pickles/adj_and_property_features.pickle', 'wb') as handle: pickle.dump(featuresDf, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print(featuresDf)

# -------------------------
### Prepare small dataset

# with open('pickles/edges.pickle', 'rb') as handle: data = pickle.load(handle)
# print(data)
# data = data[(data['From'] < 100) & (data['To'] < 100)]

# for i in range(100):
#     data = data.append({'From': i, 'To': i}, ignore_index = True)
# print(data)
# with open('pickles/playGraph_edges.pickle', 'wb') as handle: pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('pickles/attributes.pickle', 'rb') as handle: data = pickle.load(handle)
# print(data)
# data = data[0:100]
# print(data)
# with open('pickles/playGraph_attributes.pickle', 'wb') as handle: pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('pickles/playGraph_edges.pickle', 'rb') as handle: data = pickle.load(handle)
print(data)

with open('pickles/playGraph_attributes.pickle', 'rb') as handle: data = pickle.load(handle)
print(data)

# -------------------------