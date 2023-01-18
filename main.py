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
# dataset_attributes = 'attributes'         # 'attributes' or 'playGraph_attributes'
# dataset_edges = 'edges'                   # 'edges' or 'playgraph_edges'
dataset = 'UNC28'                           # 'playgraph' or 'UNC28' or 'American75'
prediction_type = 'classification'                  # 'classification' or 'regression' 
model = 'RandomForest_hyper'                                       # 'SVM' or 'RandomForest' or 'RandomForest_hyper'
# predicting_attribute = 'Status'                     # 'Status' or 'Gender' or 'Major' or 'Minor' or 'Dorm' or 'Graduation_Year' or 'High_school'
predicting_attribute = 'student_fac'                # 'student_fac' or 'gender' or 'major_index' or 'second_major' or 'dorm' or 'year' or 'high_school
selected_features = '7'                             # '1' or '2' or '3' or '4' or '5' or '6' or '7'
# ----------------------------------------


# ----------------------------------------
featuresDf, y = get_settings(dataset, predicting_attribute, prediction_type, selected_features)
print("Featurs and y collected ___________________________ ")
# ----------------------------------------
result = predict_attribute(featuresDf, y, model, prediction_type)
# print('result: ', result)
# ----------------------------------------
