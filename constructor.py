# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import csr_matrix
import pickle
import os
from nodePropertyGenerator import getNodeProperties



def load_input_data(dataset_attributes, dataset_edges):
    # read edges and attributes from pickle (full dataset)
    attr = 'pickles/' + dataset_attributes + '.pickle'
    edges = 'pickles/' + dataset_edges + '.pickle'
    
    with open(edges, 'rb') as handle: data = pickle.load(handle)
    with open(attr, 'rb') as handle: y = pickle.load(handle) 

    return data, y


# Generate the features according to criteria
def get_features(G, selected_features):
    if selected_features == '1':
        nodePropertiesDf = getNodeProperties(G)

        adjDf = nx.to_pandas_adjacency(G)
        featuresDf = pd.concat([adjDf, nodePropertiesDf], axis=1)

        featuresDf.columns = featuresDf.columns.astype(str)
        
    return featuresDf


def get_settings(dataset_attributes, dataset_edges, model, predicting_attribute, prediction_type, selected_features):
    data, y = load_input_data(dataset_attributes, dataset_edges)
    print(data)
    G = nx.from_pandas_edgelist(data, 'From', 'To')

    y = y[predicting_attribute]

    featuresDf = get_features(G, selected_features)

    print(y.shape)
    print(featuresDf.shape)

    return featuresDf, y





