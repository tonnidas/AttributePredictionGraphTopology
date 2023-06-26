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
def get_features(G, selected_features, dataset, predicting_attribute, embedding):
    graph_file = 'pickles/generated_nodeProperties/nodeProperties_' + dataset + '.pickle'
    emb_file = 'pickles/generated_embeddings/' + embedding + '_' + dataset + '.pickle'
    adjDf = nx.to_pandas_adjacency(G)
    feature_orig = pd.DataFrame.from_dict(G.nodes, orient='index')
    # Drop high_school column and one-hot encode other feature except predicting_attribute
    feature_orig = feature_orig.drop('high_school', axis=1).reset_index(drop=True) 
    features_list = ['student_fac', 'gender', 'major_index', 'second_major', 'dorm']
    for each_feature in features_list:
        if each_feature != predicting_attribute:
            oneHot = pd.get_dummies(feature_orig[each_feature], prefix = each_feature) # one hot encode chromosome
            feature_orig = feature_orig.join(oneHot)
            feature_orig = feature_orig.drop(each_feature, axis = 1)
    # print(feature_orig)
    # exit(0)
    with open(graph_file, 'rb') as handle: nodePropertiesDf = pickle.load(handle)
    # with open(emb_file, 'rb') as handle: embDf = pickle.load(handle) 

    # one hot encoding the features (except high_school as it has 2882 labels) ---------------------------
    featuresDf = feature_orig
    featuresDf.columns = featuresDf.columns.astype(str)
    print('featuresDf', featuresDf)
    # This onehot encoding codes will only work for a dataset from 'Facebook100'
    if predicting_attribute == 'student_fac': features_list = ['gender', 'major_index', 'second_major', 'dorm', 'year']
    if predicting_attribute == 'gender': features_list = ['student_fac', 'major_index', 'second_major', 'dorm', 'year']
    f = featuresDf.loc[:,predicting_attribute]
    for each in features_list:
        d_dict = nx.get_node_attributes(G, each)
        df = pd.DataFrame({each: list(d_dict.values())})
        df = pd.get_dummies(df[each], prefix=each)
        f = pd.concat([f, df.set_index(f.index)], axis=1)
    feature_orig = f.drop(columns=[predicting_attribute])
    print(feature_orig)
    # ---------------------------

    # Properties + adjacency
    if selected_features == 'pro_adj':  # 1
        # with open(graph_file, 'rb') as handle: nodePropertiesDf = pickle.load(handle)  
        # adjDf = nx.to_pandas_adjacency(G)
        featuresDf = pd.concat([adjDf, nodePropertiesDf.set_index(adjDf.index)], axis=1)
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Features + adjacency
    elif selected_features == 'fea_adj':  # 2
        # feature_orig = pd.DataFrame.from_dict(G.nodes, orient='index')                # To convert features from dictionary type to pandas dataframe
        # adjDf = nx.to_pandas_adjacency(G)
        featuresDf = pd.concat([feature_orig, adjDf.set_index(feature_orig.index)], axis=1)
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Features + adjacency + properties
    elif selected_features == 'fea_adj_pro':  # 3
        # with open(graph_file, 'rb') as handle: nodePropertiesDf = pickle.load(handle) 
        # feature_orig = pd.DataFrame.from_dict(G.nodes, orient='index')                # To convert features from dictionary type to pandas dataframe
        # adjDf = nx.to_pandas_adjacency(G)
        temp = pd.concat([feature_orig, adjDf.set_index(feature_orig.index)], axis=1)
        featuresDf = pd.concat([temp, nodePropertiesDf.set_index(temp.index)], axis=1)
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Properties + Embedding
    elif selected_features == 'pro_emb':  # 4
        # with open(graph_file, 'rb') as handle: nodePropertiesDf = pickle.load(handle) 
        # with open(emb_file, 'rb') as handle: embDf = pickle.load(handle) 
        featuresDf = pd.concat([nodePropertiesDf, embDf.set_index(nodePropertiesDf.index)], axis=1)
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Features + Embedding
    elif selected_features == 'fea_emb':  # 5
        # feature_orig = pd.DataFrame.from_dict(G.nodes, orient='index')                # To convert features from dictionary type to pandas dataframe
        # with open(emb_file, 'rb') as handle: embDf = pickle.load(handle) 
        featuresDf = pd.concat([feature_orig, embDf.set_index(feature_orig.index)], axis=1)
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Features + Embedding + Property
    elif selected_features == 'fea_emb_pro':  # 6
        # feature_orig = pd.DataFrame.from_dict(G.nodes, orient='index')                # To convert features from dictionary type to pandas dataframe
        # with open(emb_file, 'rb') as handle: embDf = pickle.load(handle) 
        # with open(graph_file, 'rb') as handle: nodePropertiesDf = pickle.load(handle) 
        temp = pd.concat([feature_orig, embDf.set_index(feature_orig.index)], axis=1)
        featuresDf = pd.concat([temp, nodePropertiesDf.set_index(temp.index)], axis=1)
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Properties (only)
    elif selected_features == 'pro':   # 7
        # with open(graph_file, 'rb') as handle: nodePropertiesDf = pickle.load(handle)  
        featuresDf = nodePropertiesDf
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Properties + features
    elif selected_features == 'pro_fea':  # 8
        # with open(graph_file, 'rb') as handle: nodePropertiesDf = pickle.load(handle) 
        nodePropertiesDf = nodePropertiesDf.sort_index(ascending=True)
        # feature_orig = pd.DataFrame.from_dict(G.nodes, orient='index')                # To convert features from dictionary type to pandas dataframe
        # print('feature_orig', feature_orig.shape, feature_orig)
        # print('nodePropertiesDf', nodePropertiesDf.shape, nodePropertiesDf) 
        featuresDf = pd.concat([nodePropertiesDf, feature_orig.set_index(nodePropertiesDf.index)], axis=1)
        featuresDf.columns = featuresDf.columns.astype(str)
        # featuresDf = featuresDf.drop(columns=[predicting_attribute])
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Features
    elif selected_features == 'fea':   # 9
        # featuresDf = pd.DataFrame.from_dict(G.nodes, orient='index')                # To convert features from dictionary type to pandas dataframe
        featuresDf = feature_orig
        featuresDf.columns = featuresDf.columns.astype(str)
        # featuresDf = featuresDf.drop(columns=[predicting_attribute])
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Adjacency 
    elif selected_features == 'adj':  # 10
        # adjDf = nx.to_pandas_adjacency(G)
        featuresDf = adjDf
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    # Embedding
    elif selected_features == 'emb':  # 11
        # below line for special case
        with open('pickles/generated_embeddings/{}_{}_only_adj.pickle'.format('GCN', 'Bowdoin47'), 'rb') as handle: embDf = pickle.load(handle) 
        featuresDf = embDf
        featuresDf.columns = featuresDf.columns.astype(str)
        print('featuresDf', featuresDf.shape)
        return featuresDf

    else: 
        print('Invalid selected_feature (', selected_features, ').')
        print('selected_feature can only be pro_adj, fea_adj, fea_adj_pro, pro_emb, fea_emb, fea_emb_pro, pro, pro_fea, fea, adj, emb')
        exit(1)


# Params:
#   G                 = networkx Graph
#   selected_features = parameter indicating the features to create

# Return values:
#   featuresDf        = created feature dataframe
#   y                 = the attribute to be predicted

# def get_settings(dataset_attributes, dataset_edges, model, predicting_attribute, prediction_type, selected_features):
def get_settings(dataset, predicting_attribute, prediction_type, selected_features, embedding):
    graph_file = '../graph-data/Facebook100/Raw/' + dataset + '.graphml'
    G = nx.read_graphml(graph_file)                                                             # Read the graph from 'Facebook100' folder
    featuresDf = get_features(G, selected_features, dataset, predicting_attribute, embedding)   # Get generated node properties along with other required properties of the graph G
    # yDF = pd.DataFrame.from_dict(G.nodes, orient='index')                                       # Convert the dictionary of features to pandas dataframe
    print('get settings er = ', featuresDf.shape)
    # special
    with open('pickles/generated_y_Df/{}_{}_yDf_student_fac.pickle'.format('GCN', 'Bowdoin47'), 'rb') as handle: yDF = pickle.load(handle) 
    y = yDF
    # y = yDF[predicting_attribute]                                                               # get the attribute to be predicted
    return featuresDf, y