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
#   G                    = networkx Graph
#   selected_features    = parameter indicating the features to create
#   dataset              = name of the dataset
#   predicting_attribute = the attribute to be predicted
# Return values:
#   featuresDf = created feature dataframe
# Generate the features according to criteria of 'selected_features'
def get_features(G, selected_features, dataset, predicting_attribute):
    graph_file = 'pickles/generated_nodeProperties/nodeProperties_' + dataset + '.pickle'
    emb_file = 'pickles/generated_embeddings/embedding_' + dataset + '.pickle'

    # Properties + adjacency
    if selected_features == '1':
        # nodePropertiesDf = getNodeProperties(G)
        with open(graph_file, 'rb') as handle: nodePropertiesDf = pickle.load(handle)  
        adjDf = nx.to_pandas_adjacency(G)
        featuresDf = pd.concat([adjDf, nodePropertiesDf.set_index(adjDf.index)], axis=1)
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Features + adjacency
    if selected_features == '2':
        feature_orig = pd.DataFrame.from_dict(G.nodes, orient='index')                # To convert features from dictionary type to pandas dataframe
        adjDf = nx.to_pandas_adjacency(G)
        featuresDf = pd.concat([feature_orig, adjDf.set_index(feature_orig.index)], axis=1)
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Features + adjacency + properties
    if selected_features == '3':
        with open(graph_file, 'rb') as handle: nodePropertiesDf = pickle.load(handle) 
        feature_orig = pd.DataFrame.from_dict(G.nodes, orient='index')                # To convert features from dictionary type to pandas dataframe
        adjDf = nx.to_pandas_adjacency(G)
        temp = pd.concat([feature_orig, adjDf.set_index(feature_orig.index)], axis=1)
        featuresDf = pd.concat([temp, nodePropertiesDf.set_index(temp.index)], axis=1)
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Properties + Embedding
    if selected_features == '4':
        with open(graph_file, 'rb') as handle: nodePropertiesDf = pickle.load(handle) 
        with open(emb_file, 'rb') as handle: embDf = pickle.load(handle) 
        featuresDf = pd.concat([nodePropertiesDf, embDf.set_index(nodePropertiesDf.index)], axis=1)
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Features + Embedding
    if selected_features == '5':
        feature_orig = pd.DataFrame.from_dict(G.nodes, orient='index')                # To convert features from dictionary type to pandas dataframe
        with open(emb_file, 'rb') as handle: embDf = pickle.load(handle) 
        featuresDf = pd.concat([feature_orig, embDf.set_index(feature_orig.index)], axis=1)
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Features + Embedding + Property
    if selected_features == '6':
        feature_orig = pd.DataFrame.from_dict(G.nodes, orient='index')                # To convert features from dictionary type to pandas dataframe
        with open(emb_file, 'rb') as handle: embDf = pickle.load(handle) 
        with open(graph_file, 'rb') as handle: nodePropertiesDf = pickle.load(handle) 
        temp = pd.concat([feature_orig, embDf.set_index(feature_orig.index)])
        featuresDf = pd.concat([temp, nodePropertiesDf.set_index(temp.index)], axis=1)
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Properties (only)
    if selected_features == '7':
        # nodePropertiesDf = getNodeProperties(G)
        with open(graph_file, 'rb') as handle: nodePropertiesDf = pickle.load(handle)  
        featuresDf = nodePropertiesDf
        return featuresDf

    # Properties + features
    if selected_features == '8':
        with open(graph_file, 'rb') as handle: nodePropertiesDf = pickle.load(handle) 
        nodePropertiesDf = nodePropertiesDf.sort_index(ascending=True)
        feature_orig = pd.DataFrame.from_dict(G.nodes, orient='index')                # To convert features from dictionary type to pandas dataframe
        print('feature_orig', feature_orig.shape, feature_orig)
        print('nodePropertiesDf', nodePropertiesDf.shape, nodePropertiesDf) 
        featuresDf = pd.concat([nodePropertiesDf, feature_orig.set_index(nodePropertiesDf.index)], axis=1)
        print('cancated = ', featuresDf.shape, featuresDf)
        featuresDf.columns = featuresDf.columns.astype(str)
        featuresDf = featuresDf.drop(columns=[predicting_attribute])
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Features
    if selected_features == '9':
        featuresDf = pd.DataFrame.from_dict(G.nodes, orient='index')                # To convert features from dictionary type to pandas dataframe
        featuresDf.columns = featuresDf.columns.astype(str)
        featuresDf = featuresDf.drop(columns=[predicting_attribute])
        print('featuresDf', featuresDf.shape)
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
    G = nx.read_graphml(graph_file)                                                             # Read the graph from 'Facebook100' folder
    featuresDf = get_features(G, selected_features, dataset, predicting_attribute)              # Get generated node properties along with other required properties of the graph G
    yDF = pd.DataFrame.from_dict(G.nodes, orient='index')                                       # Convert the dictionary of features to pandas dataframe
    y = yDF[predicting_attribute]                                                               # get the attribute to be predicted
    return featuresDf, y