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


with open('pickles/generated_nodeProperties/nodeProperties_Amherst41.pickle', 'rb') as handle: nodePropertiesDf = pickle.load(handle) 
print(nodePropertiesDf)