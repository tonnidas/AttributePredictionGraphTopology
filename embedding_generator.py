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

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dataset')
parser.add_argument('--embedding')

args = parser.parse_args()
print('Arguments:', args)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------


data_name = args.dataset   # 'American75'
embedding = args.embedding # 'Node2vec'


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
print('Generate ' + embedding + ' embedding for dataset = ' + data_name + '.')

graph_file = 'Facebook100/fb100/' + data_name + '.graphml'
nxGraph = nx.read_graphml(graph_file)  # Read the graph from 'Facebook100' folder    
nodes_list = list(nxGraph.nodes)

# make StellarGraph from nx graph
sgGraph = StellarGraph.from_networkx(nxGraph, node_type_default="people", edge_type_default="friendship", node_features="feature")
outputDf = run(embedding, data_name, nodes_list, data_name, sgGraph, 42)

outputFileName = "Result/Embedding_scores/{}_{}_roc_auc.txt".format(embedding, data_name)
f1 = open(outputFileName, "w")
f1.write("For data_name: {}, split: {}, hop: {} \n".format(data_name, 42, 0))
f1.write(outputDf.to_string())
f1.close()
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------