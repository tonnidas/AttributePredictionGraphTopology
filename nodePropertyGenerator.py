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
# Params:
#   G = networkx graph
# Return values:
#   features = six generated topological features (concatenated in a dataframe)
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


    print('done calculating node properties')
    features = pd.concat([df1, df2, df3, df4, df5, df6], axis=1)
    return features
# ----------------------------------------


# ----------------------------------------
# Params:
#   G = networkx graph
# Return values:
#   features = six generated topological features (concatenated in a dataframe)
# Generate current_flow_closeness_centrality, current_flow_betweenness_centrality, eigenvector_centrality, Katz_centrality, communi_betweenness_centrality, load_centrality of G
def getSecondPhaseNodeProperties(G):
    print('calculating associated_cliques')
    clique_dict =  nx.cliques_containing_node(G)
    newClique_dict = {key: len(value) for key, value in clique_dict.items()}
    # print("no of cliques = ", newc, type(newc))
    df_clique_no = pd.DataFrame.from_dict(newClique_dict, orient='index', columns=['associated_cliques'])

    print("calculating associated max clique length")
    df_clique_size = pd.DataFrame.from_dict(nx.node_clique_number(G), orient='index', columns=['asso_max_clique_size'])

    # print('calculating current_flow_closeness_centrality')
    # df7 = pd.DataFrame.from_dict(nx.current_flow_closeness_centrality(G), orient='index', columns=['current_flow_closeness_centrality'])

    # print('calculating current_flow_betweenness_centrality')
    # df8 = pd.DataFrame.from_dict(nx.current_flow_betweenness_centrality(G), orient='index', columns=['current_flow_betweenness_centrality'])

    print('calculating eigenvector_centrality')
    df9 = pd.DataFrame.from_dict(nx.eigenvector_centrality(G), orient='index', columns=['eigenvector_centrality'])

    print('calculating Katz_centrality')
    df10 = pd.DataFrame.from_dict(nx.katz_centrality(G, 1 / max(nx.adjacency_spectrum(G)) - 0.01), orient='index', columns=['Katz_centrality'])
    df10['Katz_centrality'] = np.real(df10.Katz_centrality)

    # print('calculating communi_betweenness_centrality')
    # df11 = pd.DataFrame.from_dict(nx.communicability_betweenness_centrality(G), orient='index', columns=['communi_betweenness_centrality'])

    print('calculating load_centrality')
    df12 = pd.DataFrame.from_dict(nx.load_centrality(G), orient='index', columns=['load_centrality'])


    print('done calculating node properties')
    features = pd.concat([df_clique_no, df_clique_size, df9, df10, df12], axis=1)

    return features
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Start by mentioning name of the dataset
dataset = 'Amherst41'

# Read any graph dataset with 'graphml' extension from 'Facebook100'
read_file = 'Facebook100/fb100/{}.graphml'.format(dataset)
G = nx.read_graphml(read_file)


# Temporarily commented. Remove comment when 'generated_nodeProperties' collection done
# --------------------------------------------
# G_name = 'timepoint0_baseline_graph'
# file_pickle = '{}.pickle'.format(G_name)
# with open(file_pickle, 'rb') as handle: G = pickle.load(handle) 

# edgelist = [(1,2), (3,4), (2,4), (4,5), (3,5), (6,7)]
# G = nx.from_edgelist(edgelist)
# --------------------------------------------

# get the properties from both methods (temporary: as firstNodePropertiesDf is already calculated for some datasets)
# with open('pickles/generated_nodeProperties_6d/nodeProperties_{}.pickle'.format(dataset), 'rb') as handle: firstNodePropertiesDf = pickle.load(handle)
secondPhaseNodePropertiesDf = getSecondPhaseNodeProperties(G)
# nodePropertiesDf = pd.concat([firstNodePropertiesDf, secondPhaseNodePropertiesDf], axis=1)


# print(nodePropertiesDf)
print(secondPhaseNodePropertiesDf)

# store generated features of the dataset
# write_file = 'pickles/generated_nodeProperties/nodeProperties_{}.pickle'.format(dataset)
# with open(write_file, 'wb') as handle: pickle.dump(nodePropertiesDf, handle, protocol=pickle.HIGHEST_PROTOCOL)
# ----------------------------------------