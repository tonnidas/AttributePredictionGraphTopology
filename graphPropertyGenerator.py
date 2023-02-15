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



def getGraphProperties(graphs_list, dataset):

    graphProperties = pd.DataFrame(columns=['graph_name', 'is_connected?', 'components_number', 'total_nodes', 'dominating_nodes', 'total_edges', 'total_cycles'])

    i = 0
    for each_graph in graphs_list:
        graphProperties.loc[len(graphProperties.index)] = [dataset[i], nx.is_connected(each_graph), nx.number_connected_components(each_graph), each_graph.number_of_nodes(), len(nx.dominating_set(each_graph)), each_graph.number_of_edges(), len(nx.cycle_basis(each_graph))]
        i += 1

    graphProperties = graphProperties.set_index('graph_name')
    return graphProperties

dataset = ['American75', 'Amherst41', 'Auburn71', 'Baylor93', 'UNC28', 'BC17', 'Berkeley13', 'Bingham82', 'Bowdoin47', 'Brandeis99', 'Brown11', 'BU10']

graphs_list = []
for each in dataset:
    graph_file = 'Facebook100/fb100/' + each + '.graphml'
    graph = nx.read_graphml(graph_file) 
    graphs_list.append(graph)

graph_basics = getGraphProperties(graphs_list, dataset)

f = 'Result/' + 'Graph_basic_properties' + '.xlsx'
graph_basics.to_excel(f) 
print(graph_basics)