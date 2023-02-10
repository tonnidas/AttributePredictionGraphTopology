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
def eigen(G):
    adj = nx.adjacency_matrix(G).todense()
    print('Tonni: ', adj, adj.shape)
    matrix1 = adj
    matrix2 = np.ones(adj.shape[0])
    print(matrix2)
    # print('type(G) == nx.MultiGraph = ', type(G) == nx.MultiGraph)
    tol = 0.0000010
    max_iter = 1 # 100

    for i in range(max_iter):

        res = np.matmul(matrix1, matrix2)
 
        print (res)

# ----------------------------------------
# Params:
#   G = networkx graph
# Return values:
#   features = all generated topological features (concatenated in a dataframe)
# Generate degree, degree_centrality, clustering_coefficient, eccentricity, closeness_centrality, betweenness_centrality of G
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

    print('calculating eigenvector_centrality')
    df7 = pd.DataFrame.from_dict(nx.eigenvector_centrality(G), orient='index', columns=['eigenvector_centrality'])

    print('calculating self-defined eigenvector_centrality')
    temp = eigen(G)
    # df8 = pd.DataFrame.from_dict(temp, orient='index', columns=['eigenvector_centrality'])


    print('done calculating node properties')
    features = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=1)
    return features
# ----------------------------------------


# ----------------------------------------
# Start by mentioning name of the dataset
dataset = 'American75'

# Read any graph dataset with 'graphml' extension from 'Facebook100'
read_file = 'Facebook100/fb100/{dataset}.graphml'.format(dataset)
G = nx.read_graphml(read_file)


# Temporarily commented. Remove comment when 'generated_nodeProperties' collection done
# --------------------------------------------
# G_name = 'timepoint0_baseline_graph'
# file_pickle = '{}.pickle'.format(G_name)
# with open(file_pickle, 'rb') as handle: G = pickle.load(handle) 
# --------------------------------------------

# generate features
nodePropertiesDf = getNodeProperties(G)
print(nodePropertiesDf)


# store generated features of the dataset
write_file = 'pickles/generated_nodeProperties/nodeProperties_{dataset}.pickle'.format(dataset)
with open(write_file, 'wb') as handle: pickle.dump(nodePropertiesDf, handle, protocol=pickle.HIGHEST_PROTOCOL)
# ----------------------------------------