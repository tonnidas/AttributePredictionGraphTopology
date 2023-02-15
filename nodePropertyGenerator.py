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


# ----------------------------------------
# Params:
#   G = networkx graph
# Return values:
#   features = ten generated topological features (concatenated in a dataframe)
# Generate degree, degree_centrality, clustering_coefficient, eccentricity, closeness_centrality, betweenness_centrality, associated_cliques, asso_max_clique_size, eigenvector_centrality, load_centrality of G
def getNodeProperties(G):

    print('calculating degree')
    temp_dict = dict(G.degree)
    temp_max = max(temp_dict.values())
    degree_regularized = {key: value / temp_max for key, value in temp_dict.items()}
    df_degree = pd.DataFrame.from_dict(degree_regularized, orient='index', columns=['degree'])

    print('calculating degree_centrality')
    df_degree_centrality = pd.DataFrame.from_dict(nx.degree_centrality(G), orient='index', columns=['degree_centrality'])
    features = pd.concat([df_degree, df_degree_centrality.set_index(df_degree.index)], axis=1)

    print('calculating clustering_coefficient')
    df_cluster_coefficient = pd.DataFrame.from_dict(nx.clustering(G), orient='index', columns=['clustering_coefficient'])
    features = pd.concat([features, df_cluster_coefficient.set_index(features.index)], axis=1)

    print('calculating eccentricity')
    graphs = list(G.subgraph(c) for c in nx.connected_components(G))     # returns a list of disconnected graphs as subgraphs
    dict_1 = {}
    for subgraph in graphs: dict_1 = {**dict_1,**nx.eccentricity(subgraph)}
    ecc_regularized = {key: value / max(dict_1.values()) for key, value in dict_1.items()}
    df_ecc = pd.DataFrame.from_dict(ecc_regularized, orient='index', columns=['eccentricity'])
    features = pd.concat([features, df_ecc.set_index(features.index)], axis=1)

    print('calculating closeness_centrality')
    df_closeness = pd.DataFrame.from_dict(nx.closeness_centrality(G), orient='index', columns=['closeness_centrality'])
    features = pd.concat([features, df_closeness.set_index(features.index)], axis=1)

    print('calculating betweenness_centrality')
    df_betweenness = pd.DataFrame.from_dict(nx.betweenness_centrality(G), orient='index', columns=['betweenness_centrality'])
    features = pd.concat([features, df_betweenness.set_index(features.index)], axis=1)

    print('calculating associated_cliques')
    clique_dict =  nx.cliques_containing_node(G)
    newClique_dict = {key: len(value) for key, value in clique_dict.items()}
    df_clique_no = pd.DataFrame.from_dict(newClique_dict, orient='index', columns=['associated_cliques'])
    features = pd.concat([features, df_clique_no.set_index(features.index)], axis=1)

    print("calculating associated max clique length")
    df_clique_size = pd.DataFrame.from_dict(nx.node_clique_number(G), orient='index', columns=['asso_max_clique_size'])
    features = pd.concat([features, df_clique_size.set_index(features.index)], axis=1)

    print('calculating eigenvector_centrality')
    df_eigen = pd.DataFrame.from_dict(nx.eigenvector_centrality(G), orient='index', columns=['eigenvector_centrality'])
    features = pd.concat([features, df_eigen.set_index(features.index)], axis=1)

    print('calculating load_centrality')
    df_load = pd.DataFrame.from_dict(nx.load_centrality(G), orient='index', columns=['load_centrality'])
    features = pd.concat([features, df_load.set_index(features.index)], axis=1)

    # -------------------- ONLY FOR CONNECTED GRAPHS ---------------------------------------------------------
    # print('calculating current_flow_closeness_centrality')
    # df_cur_flow_closeness = pd.DataFrame.from_dict(nx.current_flow_closeness_centrality(G), orient='index', columns=['current_flow_closeness_centrality'])

    # print('calculating current_flow_betweenness_centrality')
    # df_cur_flow_ = pd.DataFrame.from_dict(nx.current_flow_betweenness_centrality(G), orient='index', columns=['current_flow_betweenness_centrality'])

    # print('calculating Katz_centrality')
    # df10 = pd.DataFrame.from_dict(nx.katz_centrality(G, 1 / max(nx.adjacency_spectrum(G)) - 0.11), orient='index', columns=['Katz_centrality'])
    # df10['Katz_centrality'] = np.real(df10.Katz_centrality)
    
    # print('calculating communi_betweenness_centrality')
    # df11 = pd.DataFrame.from_dict(nx.communicability_betweenness_centrality(G), orient='index', columns=['communi_betweenness_centrality'])
    # features = pd.concat([df_clique_no, df_clique_size, df9, df12], axis=1)
    # -------------------- ONLY FOR CONNECTED GRAPHS ---------------------------------------------------------

    print('done calculating node properties')

    return features
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dataset')

args = parser.parse_args()
print('Arguments:', args)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Start by mentioning name of the dataset
dataset = args.dataset # 'Baylor93'

print('Generating node properties for dataset = ' + dataset + '.')

read_file = 'Facebook100/fb100/{}.graphml'.format(dataset)        # Read any graph dataset with 'graphml' extension from 'Facebook100'
G = nx.read_graphml(read_file)                                    # G = nx.from_edgelist([(1,2), (3,4), (2,4), (4,5), (3,5), (6,7)])

# get the properties from both methods 
nodePropertiesDf = getSecondPhaseNodeProperties(G)
print('Size of the nodeproperty dataframe = ', len(nodePropertiesDf), nodePropertiesDf)

# store generated features of the dataset
write_file = 'pickles/generated_nodeProperties/nodeProperties_{}.pickle'.format(dataset)
with open(write_file, 'wb') as handle: pickle.dump(nodePropertiesDf, handle, protocol=pickle.HIGHEST_PROTOCOL)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------