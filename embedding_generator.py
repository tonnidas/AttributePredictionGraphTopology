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


data_name = 'American75'
nxGraph = nx.read_graphml(data_name + '.graphml')  # Read the graph from 'Facebook100' folder    
nodes_list = list(nxGraph.nodes)

# make StellarGraph from nx graph
sgGraph = StellarGraph.from_networkx(nxGraph, node_type_default="people", edge_type_default="friendship", node_features="feature")
outputDf = run(data_name, nodes_list, data_name, sgGraph, 42)

outputFileName = "{}_{}_hop_{}.txt".format(data_name, 42, 0)
f1 = open(outputFileName, "w")
f1.write("For data_name: {}, split: {}, hop: {} \n".format(data_name, 42, 0))
f1.write(outputDf.to_string())
f1.close()