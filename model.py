# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import csr_matrix
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from metrics import accuracyMeasurement


# ----------------------------------------
def svmClassifier(Train, Test, Train_True_Labels):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(Train, Train_True_Labels)
    predicted_labels = clf.predict(Test)
    return predicted_labels
# ----------------------------------------


def SVM(X_train, X_test, y_train, y_test):
    # ----------------------------------------
    # Accuracy finding by svm method
    print('running svm')
    predicted_labels_svm = svmClassifier(X_train, X_test, y_train)
    print('svm completed')

    # print("svm predicted labels:", predicted_labels_svm)
    # print("true labels:", y_test.tolist())

    return predicted_labels_svm
    # ----------------------------------------


def predict_attribute(featuresDf, y, model):
    X_train, X_test, y_train, y_test = train_test_split(featuresDf, y, random_state=104, test_size=0.30, shuffle=True)

    # If model == 'SVM':
    predicted_labels_svm = SVM(X_train, X_test, y_train, y_test)

    accuracy_svm = accuracyMeasurement(y_test.tolist(), predicted_labels_svm)
    # print("svm accuracy", accuracy_svm)

    return accuracy_svm

