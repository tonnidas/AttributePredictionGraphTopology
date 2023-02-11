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
from metrics import accuracyMeasurement, classification_metrics
# ----------------------------------------


# ----------------------------------------
dataset = 'Amherst41'                           # 'playgraph' or 'UNC28' or 'American75' or 'Amherst41'
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
# Params:
#   dataset                 = a string containing dataset name
#   prediction_type         = a string ('classification' or 'regression')
#   model                   = a string
#   predicting_attribute    = a string
#   selected_features       = a string
#   rand_state_for_split    = an integer
# Return values:
#   acc                     = a float
#   f1_macro                = a float
#   precision_macro         = a float
#   recall_macro            = a float
#   f1_micro                = a float
#   precision_micro         = a float
#   recall_micro            = a float
#   f1_weighted             = a float
#   adj_RI                  = a float
# def get_settings(dataset_attributes, dataset_edges, model, predicting_attribute, prediction_type, selected_features):
def do_category_specific_task_prediction(dataset, prediction_type, model, predicting_attribute, selected_features, rand_state_for_split):
    # Get features and labels
    featuresDf, y = get_settings(dataset, predicting_attribute, prediction_type, selected_features)
    print("Featurs and y collected ___________________________ ")
    # result = predict_attribute(featuresDf, y, model, prediction_type, rand_state_for_split)

    # Get labels_test and labels_predicted
    y_test, predicted_labels = predict_attribute(featuresDf, y, model, prediction_type, rand_state_for_split)

    # Get evaluation metric values
    if prediction_type == 'classification':
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, f1_weighted, adj_RI = classification_metrics(y_test.tolist(), predicted_labels)
        print("acc: ", acc, "f1_macro: ", f1_macro, "precision_macro: ", precision_macro, "recall_macro: ", recall_macro, "f1_micro: ", f1_micro, "precision_micro: ", precision_micro, "recall_micro: ", recall_micro, "f1_weighted: ", f1_weighted, "adj_RI: ", adj_RI)

    return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, f1_weighted, adj_RI
# ----------------------------------------