# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
# np.seterr(divide='ignore', invalid='ignore')    # for allowing divide by zero
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
    print('calculating degree')
    df1 = pd.DataFrame.from_dict(dict(G.degree), orient='index', columns=['degree'])

    print('calculating degree_centrality')
    df2 = pd.DataFrame.from_dict(nx.degree_centrality(G), orient='index', columns=['degree_centrality'])

    print('calculating clustering_coefficient')
    df3 = pd.DataFrame.from_dict(nx.clustering(G), orient='index', columns=['clustering_coefficient'])

    print('calculating eccentricity')
    print("Is G connected? = ", nx.is_connected(G))
    print("How many components? = ", nx.number_connected_components(G))
    # returns list of nodes in different connected components
    print(list(nx.connected_components(G)))
    # df4 = pd.DataFrame.from_dict(nx.eccentricity(G), orient='index', columns=['eccentricity'])

    print('calculating closeness_centrality')
    df5 = pd.DataFrame.from_dict(nx.closeness_centrality(G), orient='index', columns=['closeness_centrality'])

    print('calculating betweenness_centrality')
    df6 = pd.DataFrame.from_dict(nx.betweenness_centrality(G), orient='index', columns=['betweenness_centrality'])

    print('done calculating node properties')

    # return pd.concat([df1, df2, df3, df4, df5, df6], axis=1)
    return pd.concat([df1, df2, df3, df5, df6], axis=1)
# ----------------------------------------

# read edges and attributes from pickle (full dataset)
# with open('pickles/edges.pickle', 'rb') as handle: data = pickle.load(handle)
# with open('pickles/attributes.pickle', 'rb') as handle: y = pickle.load(handle) 

# read edges from pickle (small dataset)
# with open('pickles/playGraph_edges.pickle', 'rb') as handle: data = pickle.load(handle)  
# with open('pickles/playGraph_attributes.pickle', 'rb') as handle: y = pickle.load(handle) 

# generate features
# G = nx.from_pandas_edgelist(data, 'From', 'To')
# nodePropertiesDf = getNodeProperties(G)
# adjDf = nx.to_pandas_adjacency(G)
# featuresDf = pd.concat([adjDf, nodePropertiesDf], axis=1)

# store generated features
# with open('pickles/adj_and_property_features.pickle', 'wb') as handle: pickle.dump(featuresDf, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load generated features
# with open('pickles/adj_and_property_features.pickle', 'rb') as handle: featuresDf = pickle.load(handle) 

# y = y['Gender']
# featuresDf.columns = featuresDf.columns.astype(str)

# print(y.shape)
# print(featuresDf.shape)

# X_train, X_test, y_train, y_test = train_test_split(featuresDf, y, random_state=104, test_size=0.30, shuffle=True)

# ----------------------------------------
# Accuracy finding by svm method
# print('running svm')
# predicted_labels_svm = svmClassifier(X_train, X_test, y_train)
# print('svm completed')

# print("svm predicted labels:", predicted_labels_svm)
# print("true labels:", y_test.tolist())

# accuracy_svm = accuracyMeasurement(y_test.tolist(), predicted_labels_svm)
# print("svm accuracy", accuracy_svm)
# ----------------------------------------