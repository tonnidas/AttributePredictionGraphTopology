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
# ----------------------------------------
# Global variables
nodesList = set()
degree = list()
adj = list()
neighbors = list()
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
def getClusteringCoefficient(data, node_property_matrix):
    links_between = [0] * len(nodesList)
    for i in range(len(nodesList)):
        for each_neighbour in neighbors[i]:
            for each in neighbors[i]:
                if adj[each_neighbour][each] == 1 and each_neighbour != each: links_between[i] = links_between[i] + 1

    for i in range(len(links_between)):
        links_between[i] = int(links_between[i]) // 2

    for row in range(len(node_property_matrix)):
        temp1 = float(2 * links_between[row])
        temp2 = node_property_matrix[row][0] * (node_property_matrix[row][0] - 1)
        temp = round((temp1 / temp2), 3)
        if temp2 == 0: temp = 0.0
        node_property_matrix[row][2] = temp
    return node_property_matrix
# ----------------------------------------


# Main part starts
# ----------------------------------------
with open('pickles/playGraph_edges.pickle', 'rb') as handle: data = pickle.load(handle)                                        # read edges pickle in C
setDegree(data) # sets the global variables needed for important methods
setAdjacency(data)
setAdjacentNeighbours()
print(nodesList, degree)
node_property_matrix = np.zeros((len(nodesList),6))    # as 6 property 



node_property_matrix = getDegree(data, node_property_matrix)
node_property_matrix = getDegreeCentrality(node_property_matrix)
node_property_matrix = getClusteringCoefficient(data, node_property_matrix)
print(node_property_matrix)