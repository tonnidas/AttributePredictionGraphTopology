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
dataset = 'Amherst41'                           # 'playgraph' or 'UNC28' or 'American75'
prediction_type = 'classification'                  # 'classification' or 'regression' 
model = 'RandomForest_hyper'                              # 'SVM' or 'RandomForest' or 'RandomForest_hyper'
predicting_attribute = 'gender'                     # 'student_fac' or 'gender' or 'major_index' or 'second_major' or 'dorm' or 'year' or 'high_school  ('Status' = 'student_fac')
selected_features = '1'                             # '1' or '2' or '3' or '4' or '5' or '6' or '7' or '8'
# Categories and their meaning in selected_features:
    # '1' = Property + Adjacency  
    # '2' = Features + Adjacency
    # '3' = Features + Adjacency + Property
    # '4' = Property + Embedding
    # '5' = Features + Embedding
    # '6' = Features + Embedding + Property
    # '7' = Properties
    # '8' = Properties + Features
    # '9' = Features

rand_state_for_split = 15
# ----------------------------------------

# ----------------------------------------
    # def __init__(self, dataset, prediction_type, model, predicting_attribute, selected_features, rand_state_for_split):
    #     self.dataset = dataset
    #     self.prediction_type = prediction_type
    #     self.model = model
    #     self.predicting_attribute = predicting_attribute
    #     self.selected_features = selected_features
    #     self.rand_state_for_split = rand_state_for_split 
# ----------------------------------------

# ----------------------------------------
    # @staticmethod
def do(dataset, prediction_type, model, predicting_attribute, selected_features, rand_state_for_split):
    featuresDf, y = get_settings(dataset, predicting_attribute, prediction_type, selected_features)
    print("Featurs and y collected ___________________________ ")
    # result = predict_attribute(featuresDf, y, model, prediction_type, rand_state_for_split)

    return predict_attribute(featuresDf, y, model, prediction_type, rand_state_for_split)
# ----------------------------------------