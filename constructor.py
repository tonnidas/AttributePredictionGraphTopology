# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import csr_matrix
import pickle
import os


# Params:
#   dataset_attributes = name of the attribute file of dataset
#   dataset_edges      = name of the edges file of dataset
# Return values:
#   data = all edges
#   y    = all attributes
# Read the pickle files for edges and attributes and return them in dataframes for usage (for special case)
def load_input_data(dataset_attributes, dataset_edges):
    attr = 'pickles/' + dataset_attributes + '.pickle'            # read attributes from pickle (for special case (UNC28_attributes))
    edges = 'pickles/' + dataset_edges + '.pickle'                # read edges from pickle (for special case (UNC28_edges))
    with open(edges, 'rb') as handle: data = pickle.load(handle)
    with open(attr, 'rb') as handle: y = pickle.load(handle) 
    return data, y


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
def get_settings(dataset, predicting_attribute, prediction_type, selected_features):
    graph_file = 'Facebook100/fb100/' + dataset + '.graphml'
    G = nx.read_graphml(graph_file)                                       # Read the graph from 'Facebook100' folder
    featuresDf = get_features(G, selected_features)                       # Get generated node properties along with other required properties of the graph G
    yDF = pd.DataFrame.from_dict(G.nodes, orient='index')                 # Convert the dictionary of features to pandas dataframe
    y = yDF[predicting_attribute]                                         # get the attribute to be predicted
    return featuresDf, y