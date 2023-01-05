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
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from metrics import accuracyMeasurement, classification_metrics


# ----------------------------------------

# ----------------------------------------


# ----------------------------------------
def SVM_classifier(X_train, X_test, y_train, y_test):
    print('running svm')
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    predicted_labels_svm = clf.predict(X_test)
    print('svm completed')

    # print("svm predicted labels:", predicted_labels_svm)
    # print("true labels:", y_test.tolist())

    return predicted_labels_svm
# ----------------------------------------


# ----------------------------------------
def randomForest_classifier(X_train, X_test, y_train, y_test):
    print('running random forest')
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    predicted_labels_randomForest = clf.predict(X_test)
    print('randomForest completed')

    # print("svm predicted labels:", predicted_labels_svm)
    # print("true labels:", y_test.tolist())

    return predicted_labels_randomForest
# ----------------------------------------


def predict_attribute(featuresDf, y, model, prediction_type):
    X_train, X_test, y_train, y_test = train_test_split(featuresDf, y, random_state=104, test_size=0.30, shuffle=True)

    predicted_labels = [0]
    if model == 'SVM':
        predicted_labels = SVM_classifier(X_train, X_test, y_train, y_test)
    if model == 'RandomForest':
        predicted_labels = randomForest_classifier(X_train, X_test, y_train, y_test)

    if prediction_type == 'classification':
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, f1_weighted, adj_RI = classification_metrics(y_test.tolist(), predicted_labels)
    print("acc: ", acc, "f1_macro: ", f1_macro, "precision_macro: ", precision_macro, "recall_macro: ", recall_macro, "f1_micro: ", f1_micro, "precision_micro: ", precision_micro, "recall_micro: ", recall_micro, "f1_weighted: ", f1_weighted, "adj_RI: ", adj_RI)

    return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, f1_weighted, adj_RI

