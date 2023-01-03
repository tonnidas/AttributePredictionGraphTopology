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
# Global variables
nodesList = set()
degree = list()
adj = list()
neighbors = list()
G = nx.Graph([(0, 1), (0, 2), (0, 3)])
# ----------------------------------------


# Helper methods to set the global variables
# ----------------------------------------

# Set the nodesList from the list of edges
def setNodesList(data):
    for row in range(len(data)):
        for column in range(2):
            nodesList.add(int(data[row][column]))


# Set the degree list from the list of edges
def setDegree(data):
    setNodesList(data)                         # calling setNodeList method here
    global degree
    degree = [0] * len(nodesList)
    for row in range(len(data)):
        for column in range(2):
            node = int(data[row][column]) - 1                        # as the matrix starts from 0 but the nodes number starts from 1
            degree[node] = degree[node] + 1  


# Set the adjacency matrix (bi-directional; n[0][1] = n[1][0] = 1) from a list of edges
def setAdjacency(edgeList):
    global adj
    adj = np.zeros((len(nodesList),len(nodesList)))
    for row in range(len(edgeList)):
        node1, node2 = int(edgeList[row][0]) - 1, int(edgeList[row][1]) - 1
        adj[node1][node2], adj[node2][node1] = 1, 1
    return adj


# Set the adjacency neighbours of each node from adjacency matrix
def setAdjacentNeighbours():
    global neighbors
    neighbors = [None] * len(adj)
    for i in range(len(adj)):
        neighbors[i] = set()
        for j in range(len(adj)):
            if adj[i][j] == 1: neighbors[i].add(j)


# Set the graph from the edges list
def setGraph(data):
    global G
    G = nx.from_edgelist(data)

# Get the eccentricity of a node (max length of the shortest paths from the node to the other nodes in the graph)
def get_eccentricity(adjacency: Union[sparse.csr_matrix, np.ndarray], node: int,
                     unweighted: bool = False,
                     n_jobs: Optional[int] = None) -> int:
    """
    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    node:
        The node to compute the eccentricity for.
    unweighted:
        Whether or not the graph is unweighted.
    n_jobs :
        If an integer value is given, denotes the number of workers to use (-1
        means the maximum number will be used).
        If ``None``, no parallel computations are made.

    Returns
    -------
    eccentricity : int
    """

    dists = get_distances(adjacency, node, method='D', return_predecessors=False, unweighted=unweighted, n_jobs=n_jobs).astype(int)
    return dists.max()
# ----------------------------------------



# ----------------------------------------
# Get the degree(indegree + outdegree) of each node
def getDegree(data, node_property_matrix):
    for row in range(len(data)):
        for column in range(2):
            node = int(data[row][column]) - 1                                    # as the matrix starts from 0 but the nodes number starts from 1
            node_property_matrix[node][0] = node_property_matrix[node][0] + 1    # Filling node property matrix with degree
    return node_property_matrix


# Get the degree centrality of each node
def getDegreeCentrality(node_property_matrix):
    for row in range(len(node_property_matrix)):
        temp = round(node_property_matrix[row][0] / (node_property_matrix.shape[0] - 1), 3)
        node_property_matrix[row][1] = temp
    return node_property_matrix


# Get the clustering coefficient of each node (self loop not considered)
def getClusteringCoefficient(node_property_matrix):
    temp = nx.clustering(G)
    for each in temp:
        node_property_matrix[int(each) - 1][2] = round(temp[each], 3)
    return node_property_matrix


# Get the eccentricity of each node
def getEccentricity(node_property_matrix):
    for row in range(len(node_property_matrix)):
        node_property_matrix[row][3] = get_eccentricity(adj, row, True)
    return node_property_matrix


# Get the closeness centrality of each node
def getClosenessCentrality(node_property_matrix):
    temp = nx.closeness_centrality(G)
    for each in temp:
        node_property_matrix[int(each) - 1][4] = round(temp[each], 3)
    return node_property_matrix


# Get the betweenness centrality of each node
def getBetweennessCentrality(node_property_matrix):
    temp = nx.betweenness_centrality(G)
    for each in temp:
        node_property_matrix[int(each) - 1][5] = round(temp[each], 3)
    return node_property_matrix
# ----------------------------------------


# Main part starts
# ----------------------------------------
with open('pickles/edges.pickle', 'rb') as handle: data = pickle.load(handle)   # read edges pickle in C
# print(data)
# setGraph(data)
G = nx.from_pandas_edgelist(data, 'From', 'To')
# print(len(G.nodes))

setDegree(data) # sets the global variables needed for important methods
setAdjacency(data)
setAdjacentNeighbours()
# print(nodesList, degree)
node_property_matrix = np.zeros((len(nodesList),6))    # as 6 property 



node_property_matrix = getDegree(data, node_property_matrix)
node_property_matrix = getDegreeCentrality(node_property_matrix)
node_property_matrix = getClusteringCoefficient(node_property_matrix)
node_property_matrix = getEccentricity(node_property_matrix)
node_property_matrix = getClosenessCentrality(node_property_matrix)
node_property_matrix = getBetweennessCentrality(node_property_matrix)
# print(node_property_matrix)

# print(node_property_matrix.shape)
# print(adj.shape)

features = np.hstack((node_property_matrix, adj))
# print(features.shape)


with open('pickles/attributes.pickle', 'rb') as handle: attribute = pickle.load(handle) 
print(attribute.shape)
y = [2, 3, 2, 2, 2, 2, 2, 1]
X_train, X_test, y_train, y_test = train_test_split(features, y, random_state=104, test_size=0.30, shuffle=True)

# ----------------------------------------
# Accuracy finding by svm method
predicted_labels_svm = svmClassifier(X_train, X_test, y_train)
# print("svm predicted labels: ", predicted_labels_svm, "True labels: ", Test_True_Labels)
accuracy_svm = accuracyMeasurement(y_test, predicted_labels_svm)
print("svm accuracy", accuracy_svm)
# ----------------------------------------
# print(nx.clustering(G))
# print(G.edges)