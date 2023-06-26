# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
import scipy.sparse 
from scipy.sparse import csr_matrix
import pickle
import stellargraph as sg
import os
from stellargraph import StellarGraph, datasets
from math import isclose
from sklearn.decomposition import PCA
from embedding_4_models import run



# ==================================================================================================================================================================
# Make the graph from the features and adj
def get_sg_graph(adj, features):
    print('adj shape:', adj.shape, 'feature shape:', features.shape)
    nxGraph = nx.from_scipy_sparse_array(adj)                           # make nx graph from scipy matrix

    # add features to nx graph
    for node_id, node_data in nxGraph.nodes(data=True):
        node_feature = features[node_id].todense()
        node_data["feature"] = np.squeeze(np.asarray(node_feature)) # convert to 1D matrix to array

    # make StellarGraph from nx graph
    sgGraph = StellarGraph.from_networkx(nxGraph, node_type_default="gene", edge_type_default="connects to", node_features="feature")
    print(sgGraph.info())

    return sgGraph
# ==================================================================================================================================================================

# ==================================================================================================================================================================
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--folder')
parser.add_argument('--dataset')
parser.add_argument('--embedding')

args = parser.parse_args()
print('Arguments:', args)

folder_name = args.folder  # 'Facebook100'
data_name = args.dataset   # 'American75', 'Bowdoin47'
embedding = args.embedding # 'Node2vec', 'GCN', 'Attri2Vec', 'GraphSAGE'
# python embedding_generator.py --folder=Facebook100 --dataset=Bowdoin47 --embedding=GraphSAGE
predicting_attribute = 'student_fac'     # 'gender', 'student_fac'
type = 'only_adj'                  # 'features_except_gender', 'properties_without_features', 'only_adj'
# ==================================================================================================================================================================



# read adj and features from pickle and prepare sg graph
with open('../graph-data/{}/Processed/{}_featuresDf_hop_{}.pickle'.format(folder_name, data_name, str(0)), 'rb') as handle: features = pickle.load(handle) 
with open('../graph-data/{}/Processed/{}_adjDf_hop_{}.pickle'.format(folder_name, data_name, str(0)), 'rb') as handle: adj = pickle.load(handle)
with open('pickles/generated_nodeProperties/nodeProperties_{}_special.pickle'.format(data_name), 'rb') as handle: node = pickle.load(handle)




if type == 'features_except_gender':
    features['year'] = pd.to_numeric(features['year'], errors='coerce')     # convert pval to float and use NaN for non numeric values

    # filter rows with unavailable 'year' values 
    to_be_filtered = features.index[features['year'] == 0].tolist()
    features = features.loc[(features['year'] > 0)]


    # one hot encode all features
    features = features.drop('high_school', axis=1)
    features_list = ['student_fac', 'gender', 'major_index', 'second_major', 'dorm']
    for each_feature in features_list:
        if each_feature != predicting_attribute:
            oneHot = pd.get_dummies(features[each_feature], prefix = each_feature) # one hot encode the attribute
            features = features.join(oneHot)
            features = features.drop(each_feature, axis = 1)
    features = features.reset_index(drop=True)
    print(features)

    predicting_attr = features[predicting_attribute]
    features = features.drop(predicting_attribute, axis=1)

    features = csr_matrix(features.to_numpy())


    for i in to_be_filtered: 
        adj = adj.drop(i, axis=1)
        adj = adj.drop(i)
    adj = adj.reset_index(drop=True)
        

    adj = csr_matrix(adj.to_numpy())
    print(adj)
    # exit(0)

    # make StellarGraph and list of nodes
    sgGraph = get_sg_graph(adj, features)        # make the graph
    nodes_list = list(range(0, features.shape[0]))

    with open("pickles/generated_y_Df/{}_{}_yDf_{}.pickle".format(embedding, data_name, predicting_attribute), 'wb') as handle: pickle.dump(predicting_attr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    outputDf = run(embedding, data_name, nodes_list, data_name, sgGraph, 42)
    outputFileName = "Result/Embedding_scores/{}_{}_roc_auc_except_{}.txt".format(embedding, data_name, predicting_attribute)
    f1 = open(outputFileName, "w")
    f1.write("For data_name: {}, split: {}, hop: {} \n".format(data_name, 42, 0))
    f1.write(outputDf.to_string())
    f1.close()


elif type == 'only_adj':
    features['year'] = pd.to_numeric(features['year'], errors='coerce')     # convert pval to float and use NaN for non numeric values

    # filter rows with unavailable 'year' values 
    to_be_filtered = features.index[features['year'] == 0].tolist()
    features = features.loc[(features['year'] > 0)]


    # one hot encode all features
    features = features.drop('high_school', axis=1)
    features_list = ['student_fac', 'gender', 'major_index', 'second_major', 'dorm']
    for each_feature in features_list:
        if each_feature != predicting_attribute:
            oneHot = pd.get_dummies(features[each_feature], prefix = each_feature) # one hot encode the attribute
            features = features.join(oneHot)
            features = features.drop(each_feature, axis = 1)
    features = features.reset_index(drop=True)
    print(features)

    predicting_attr = features[predicting_attribute]
    features = features.drop(predicting_attribute, axis=1)

    features = csr_matrix(features.to_numpy())


    for i in to_be_filtered: 
        adj = adj.drop(i, axis=1)
        adj = adj.drop(i)
    adj = adj.reset_index(drop=True)
        

    adj = csr_matrix(adj.to_numpy())
    print(adj)
    s = np.ones((adj.shape[0], 2))
    features = csr_matrix(s)
    print(features)
    # exit(0)

    # make StellarGraph and list of nodes
    sgGraph = get_sg_graph(adj, features)        # make the graph
    nodes_list = list(range(0, features.shape[0]))

    with open("pickles/generated_y_Df/{}_{}_yDf_{}.pickle".format(embedding, data_name, predicting_attribute), 'wb') as handle: pickle.dump(predicting_attr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    outputDf = run(embedding, data_name, nodes_list, data_name, sgGraph, 42)
    outputFileName = "Embedding_scores/{}_{}_roc_auc_onlyAdj_{}.txt".format(embedding, data_name, predicting_attribute)
    f1 = open(outputFileName, "w")
    f1.write("For data_name: {}, split: {}, hop: {} \n".format(data_name, 42, 0))
    f1.write(outputDf.to_string())
    f1.close()

else:  # type == 'properties_without_features'

    features['year'] = pd.to_numeric(features['year'], errors='coerce')     # convert pval to float and use NaN for non numeric values

    # filter rows with unavailable 'year' values 
    to_be_filtered = features.index[features['year'] == 0].tolist()
    for i in to_be_filtered: 
        adj = adj.drop(i, axis=1)
        adj = adj.drop(i)
    adj = adj.reset_index(drop=True)
        
   
    adj = csr_matrix(adj.to_numpy())
    node = csr_matrix(node.to_numpy())
    # print(adj)
    # print(node)
    # exit(0)

    # make StellarGraph and list of nodes
    sgGraph = get_sg_graph(adj, node)        # make the graph
    nodes_list = list(range(0, adj.shape[0]))

    # with open("pickles/generated_y_Df/{}_{}_roc_auc_nodeProperty_{}.txt".format(embedding, data_name, predicting_attribute), 'wb') as handle: pickle.dump(predicting_attr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    outputDf = run(embedding, data_name, nodes_list, data_name, sgGraph, 42)
    outputFileName = "Embedding_scores/{}_{}_roc_auc_nodeProperty_{}.txt".format(embedding, data_name, predicting_attribute)
    f1 = open(outputFileName, "w")
    f1.write("For data_name: {}, split: {}, hop: {} \n".format(data_name, 42, 0))
    f1.write(outputDf.to_string())
    f1.close()