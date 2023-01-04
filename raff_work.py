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


# for i in range(7):

#     # Loading attributes from a pickle
#     with open(f'z_score_normalized_old_{i}.pickle', 'rb') as handle: A = pickle.load(handle)
#     with open(f'z_score_normalized_new_{i}.pickle', 'rb') as handle: B = pickle.load(handle)

#     print((A==B).all())

def getAdjacency(edgeList, nodes):
    adj = np.zeros((len(nodes),len(nodes)))
    for row in range(len(edgeList)):
        node1 = int(edgeList[row][0]) - 1
        node2 = int(edgeList[row][1]) - 1
        adj[node1][node2] = 1
        adj[node2][node1] = 1
    print("adj", adj)
    return adj



# ----------------------------------------

# For Facebook dataset
with open('pickles/attributes.pickle', 'rb') as handle: A = pickle.load(handle)
with open('pickles/edges.pickle', 'rb') as handle: B = pickle.load(handle)
# a = scipy.sparse.csr_matrix(B.values)   # Convert the dataframe into csr matrix
# a = B.to_numpy()  # convert to matrix from dataframe
# print(B)
# print(A.shape[0])

# read csv files into one single dataframe
with open('txts/playGraph_edgeList.txt') as f: lines = f.readlines()

data = np.zeros((11,2))
i, j = 0, 0
for each in lines:
    e = each.split('\t')
    data[i][j] = int(e[0])
    temp = e[1].split('\n')
    data[i][j+1] = int(temp[0])
    i = i + 1

# print(data)

with open('pickles/playGraph_edges.pickle', 'wb') as handle: pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Main part starts
# ----------------------------------------
with open('pickles/playGraph_edges.pickle', 'rb') as handle: C = pickle.load(handle)                                        # read edges pickle in C
content_matrix = np.zeros((8,6))    # as 6 property 

all_nodes = set()

# Get the degree(indegree + outdegree) of each node
for row in range(len(data)):
    for column in range(2):
        all_nodes.add(int(data[row][column]))
        node = int(data[row][column]) - 1
        content_matrix[node][0] = content_matrix[node][0] + 1
        # print(column)


# Get the degree centrality of each node
for row in range(len(content_matrix)):
    temp = round(content_matrix[row][0] / (content_matrix.shape[0] - 1), 3)
    content_matrix[row][1] = temp


# Get the clustering coefficient of each node
adj = getAdjacency(data, all_nodes)
neighbors = [None] * len(adj)
links_between = [0] * len(adj)
for i in range(len(adj)):
    neighbors[i] = set()
    for j in range(len(adj)):
        if adj[i][j] == 1: neighbors[i].add(j)
for i in range(len(neighbors)):
    for each_neighbour in neighbors[i]:
        for each in neighbors[i]:
            if adj[each_neighbour][each] == 1 and each_neighbour != each: links_between[i] = links_between[i] + 1

for i in range(len(links_between)):
    links_between[i] = int(links_between[i]) // 2

print("link: ", links_between)

for row in range(len(content_matrix)):
    temp1 = float(2 * links_between[row])
    temp2 = content_matrix[row][0] * (content_matrix[row][0] - 1)
    print("temp1 = ", temp1, "temp2 = ", temp2)
    temp = round((temp1 / temp2), 3)
    if temp2 == 0: temp = 0.0
    content_matrix[row][2] = temp









print(content_matrix)
