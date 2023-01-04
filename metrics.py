# ----------------------------------------
# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# ----------------------------------------


# ----------------------------------------
def accuracyMeasurement(Test_True_Labels, predicted_labels):
    matched = 0
    for i in range(len(Test_True_Labels)):
        if Test_True_Labels[i] == predicted_labels[i]:
            matched = matched + 1
    accuracy = (matched / len(Test_True_Labels)) * 100
    return accuracy
# ----------------------------------------