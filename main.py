# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
from scipy.sparse import csr_matrix
import pickle
import os
from constructor import get_settings
from model import predict_attribute
# ----------------------------------------


# ----------------------------------------
dataset_attributes = 'attributes'         # 'attributes' or 'playGraph_attributes'
dataset_edges = 'edges'                   # 'edges' or 'playgraph_edges'
prediction_type = 'classification'                  # 'classification' or 'regression' 
model = 'RandomForest'                                       # 'SVM' or 'RandomForest'
predicting_attribute = 'Gender'                     # 'Status' or 'Gender' or 'Major' or 'Minor' or 'Dorm' or 'Graduation_Year' or 'High_school'
selected_features = '1'                             # '1' or '2' or '3' or '4' or '5' or '6' 
# ----------------------------------------


# ----------------------------------------
featuresDf, y = get_settings(dataset_attributes, dataset_edges, model, predicting_attribute, prediction_type, selected_features)
# ----------------------------------------
result = predict_attribute(featuresDf, y, model, prediction_type)
# print('result: ', result)
# ----------------------------------------
