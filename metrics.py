# ----------------------------------------
# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn import metrics

from munkres import Munkres, print_matrix
# ----------------------------------------


# ----------------------------------------
def accuracyMeasurement(Test_True_Labels, Test_Predicted_labels):
    matched = 0
    for i in range(len(Test_True_Labels)):
        if Test_True_Labels[i] == predicted_labels[i]:
            matched = matched + 1
    accuracy = (matched / len(Test_True_Labels)) * 100
    return accuracy
# ----------------------------------------


# ----------------------------------------
def classification_metrics(Test_True_Labels, Test_Predicted_labels):
    acc = metrics.accuracy_score(Test_True_Labels, Test_Predicted_labels)
    f1_macro = metrics.f1_score(Test_True_Labels, Test_Predicted_labels, average='macro')
    precision_macro = metrics.precision_score(Test_True_Labels, Test_Predicted_labels, average='macro')
    recall_macro = metrics.recall_score(Test_True_Labels, Test_Predicted_labels, average='macro')
    f1_micro = metrics.f1_score(Test_True_Labels, Test_Predicted_labels, average='micro')
    precision_micro = metrics.precision_score(Test_True_Labels, Test_Predicted_labels, average='micro')
    recall_micro = metrics.recall_score(Test_True_Labels, Test_Predicted_labels, average='micro')
    f1_weighted = metrics.f1_score(Test_True_Labels, Test_Predicted_labels, average='weighted')
    adj_RI = metrics.adjusted_rand_score(Test_True_Labels, Test_Predicted_labels)
    
    return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, f1_weighted, adj_RI
# ----------------------------------------