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
from helperMethods import *




# read edges from pickle (large dataset)
# with open('pickles/UNC28_edges.pickle', 'rb') as handle: data = pickle.load(handle)  
# with open('pickles/UNC28_attributes.pickle', 'rb') as handle: y = pickle.load(handle) 


# print("Status = ", y['Status'].unique(), "How many classes? = ", len(y['Status'].unique()) )
# print("Gender = ", y['Gender'].unique(), "How many classes? = ", len(y['Gender'].unique()) )
# print("Major = ", y['Major'].unique(), "How many classes? = ", len(y['Major'].unique()) )
# print("Minor = ", y['Minor'].unique(), "How many classes? = ", len(y['Minor'].unique()) )
# print("Dorm = ", y['Dorm'].unique(), "How many classes? = ", len(y['Dorm'].unique()) )
# print("Graduation_Year = ", y['Graduation_Year'].unique(), "How many classes? = ", len(y['Graduation_Year'].unique()) )
# print("High_school = ", y['High_school'].unique(), "How many classes? = ", len(y['High_school'].unique()) )


# from nodePropertyGenerator import getNodeProperties
# graph_names = ['Amherst41', 'Auburn71', 'Baylor93', 'BC17', 'Berkeley13', 'Bingham82', 'Bowdoin47', 'Brandeis99', 'Brown11', 'BU10']
# print(len(graph_names)/2)
# for i in range(0, 5, 1):
#     graph_location = 'Facebook100/fb100/' +  graph_names[i]  + '.graphml'
#     G = nx.read_graphml(graph_location)
#     print(len(G.edges), len(G.nodes))

#     nodePropertiesDf = getNodeProperties(G)

#     # # store generated features of the dataset
#     write_file = 'pickles/generated_nodeProperties/nodeProperties_' + graph_names[i] + '.pickle'
#     with open(write_file, 'wb') as handle: pickle.dump(nodePropertiesDf, handle, protocol=pickle.HIGHEST_PROTOCOL)


from main import do

graph_names = ['Amherst41', 'Auburn71', 'Baylor93', 'BC17', 'Berkeley13', 'Bingham82', 'Bowdoin47', 'Brandeis99', 'Brown11', 'BU10']
for j in range(len(graph_names)):
    dataset = graph_names[j]                           # 'playgraph' or 'UNC28' or 'American75'
    prediction_type = 'classification'                  # 'classification' or 'regression' 
    model = 'RandomForest_hyper'                              # 'SVM' or 'RandomForest' or 'RandomForest_hyper'
    predicting_attribute = 'gender'                     # 'student_fac' or 'gender' or 'major_index' or 'second_major' or 'dorm' or 'year' or 'high_school  ('Status' = 'student_fac')
    selected_features = '7'  

    abc = 'Temp_' + dataset + '_' + selected_features + '_gender.txt'
    f1 = open(abc, "w")

    rand_state_for_split = [15, 105, 199, 207, 233]
    for i in range(0, 5, 1):
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, f1_weighted, adj_RI = do(dataset, prediction_type, model, predicting_attribute, selected_features, rand_state_for_split[i])
        Line = "acc: "+str(acc)+ "f1_macro: "+ str(f1_macro)+"precision_macro: "+ str(precision_macro)+ "recall_macro: "+ str(recall_macro)+"f1_micro: "+ str(f1_micro)+ "precision_micro: "+ str(precision_micro)+ "recall_micro: "+ str(recall_micro)+ "f1_weighted: "+ str(f1_weighted)+ "adj_RI: "+ str(adj_RI)+ "\n"
        f1.write(Line)

    f1.close()

    dataset = graph_names[j]                           # 'playgraph' or 'UNC28' or 'American75'
    prediction_type = 'classification'                  # 'classification' or 'regression' 
    model = 'RandomForest_hyper'                              # 'SVM' or 'RandomForest' or 'RandomForest_hyper'
    predicting_attribute = 'student_fac'                     # 'student_fac' or 'gender' or 'major_index' or 'second_major' or 'dorm' or 'year' or 'high_school  ('Status' = 'student_fac')
    selected_features = '7'  

    abc = 'Temp_' + dataset + '_' + selected_features + '_student_fac.txt'
    f2 = open(abc, "w")

    rand_state_for_split = [15, 105, 199, 207, 233]
    for i in range(0, 5, 1):
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, f1_weighted, adj_RI = do(dataset, prediction_type, model, predicting_attribute, selected_features, rand_state_for_split[i])
        Line = "acc: "+str(acc)+ "f1_macro: "+ str(f1_macro)+"precision_macro: "+ str(precision_macro)+ "recall_macro: "+ str(recall_macro)+"f1_micro: "+ str(f1_micro)+ "precision_micro: "+ str(precision_micro)+ "recall_micro: "+ str(recall_micro)+ "f1_weighted: "+ str(f1_weighted)+ "adj_RI: "+ str(adj_RI)+ "\n"
        f2.write(Line)

    f2.close()