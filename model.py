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
# from metrics import accuracyMeasurement, classification_metrics

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import LabelEncoder
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

# ----------------------------------------
def randomForest_classifier_hyper(X_train, X_test, y_train, y_test):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 100)]

    # Number of features to consider at every split
    max_features = ['auto', 'log2', 'sqrt']

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    print('running random forest with hyperparameter')

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 5, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(X_train, y_train)

    # predict
    predicted_labels_train = rf_random.predict(X_train)
    predicted_labels_test = rf_random.predict(X_test)

    print('randomForest with hyperparameter completed')
    # summarize result
    print('Best Hyperparameters: %s' % rf_random.best_params_)

    return predicted_labels_test
# ----------------------------------------

# ----------------------------------------
def neuralNetwork_Classifier(X_train, X_test, y_train, y_test):
    encoder = LabelEncoder()                                      # encode class values as integers
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = np_utils.to_categorical(encoded_Y)                  # convert integers to dummy variables (i.e. one hot encoded)
# ----------------------------------------


# ----------------------------------------
def predict_attribute(featuresDf, y, model, prediction_type, rand_state):
    print(featuresDf.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(featuresDf, y, random_state=rand_state, test_size=0.30, shuffle=True)

    predicted_labels = [0]
    if model == 'SVM':
        predicted_labels = SVM_classifier(X_train, X_test, y_train, y_test)
    if model == 'RandomForest':
        predicted_labels = randomForest_classifier(X_train, X_test, y_train, y_test)
    if model == 'RandomForest_hyper':
        predicted_labels = randomForest_classifier_hyper(X_train, X_test, y_train, y_test)

    return y_test, predicted_labels
    # return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, f1_weighted, adj_RI
# ----------------------------------------