#!/usr/bin/python

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import re
import sys
import pprint
import operator
import scipy.stats
from time import time
import tester
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Extract original features list from the data_dict
features_list = data_dict[data_dict.keys()[0]].keys()

# Seperate the POI label from features_list for now and remove email_address
features_list.remove('poi')
features_list.remove('email_address')


### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("LOCKHART EUGENE E", 0)


### Task 3: Create new feature(s)
# Convert data_dict to pandas dataframe
df = pd.DataFrame(data_dict)
df_trans = df.transpose()

# Replace string "NaN" to zero (0)
def to_zero(v):
    if v == 'NaN':
        v = 0
    return v
df_trans = df_trans.applymap(to_zero)

# Remove column email_address from df_trans
df_trans = df_trans.drop('email_address', 1)

# Create new features of relative values of each payment feature to total_payments
payment_features = ['salary', 'bonus', 'long_term_incentive', \
                    'deferral_payments', 'loan_advances', 'other', \
                    'expenses', 'director_fees', 'deferred_income']

rel_payment = []
for feature in payment_features:
    new_feature_name = 'rel_' + feature
    df_trans[new_feature_name] = (df_trans[feature]/df_trans['total_payments']).replace([np.inf, -np.inf, np.nan], 0)
    rel_payment.append(new_feature_name)

# Add total_payment to the list of payment_features
payment_features.append('total_payments')

# Create new features of relative values of each stock feature to total_stock_value
stock_features = ['exercised_stock_options', 'restricted_stock', \
                  'restricted_stock_deferred']

rel_stock = []
for feature in stock_features:
    new_feature_name = 'rel_' + feature
    df_trans[new_feature_name] = (df_trans[feature]/df_trans['total_stock_value']).replace([np.inf, -np.inf, np.nan], 0)
    rel_stock.append(new_feature_name)

# Add total_payment to the list of stock_features
stock_features.append('total_stock_value')

# Create new features of fraction of emails exchanged with POI
df_trans['fraction_poi']=((df_trans['from_this_person_to_poi']+\
                          df_trans['from_poi_to_this_person'])/\
(df_trans['from_messages']+df_trans['to_messages'])).fillna(0)

df_trans['fraction_to_poi']=(df_trans['from_this_person_to_poi']/\
df_trans['from_messages']).fillna(0)

df_trans['fraction_from_poi']=(df_trans['from_poi_to_this_person']/\
df_trans['to_messages']).fillna(0)

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',
                     'from_this_person_to_poi', 'shared_receipt_with_poi', 
                      'fraction_poi', 'fraction_to_poi', 'fraction_from_poi']

# Create new lists of features
financial_features = payment_features+stock_features
rel_financial_features = rel_payment+rel_stock
total_features = financial_features + email_features
rel_total_features = rel_financial_features + email_features

### Feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_trans), \
                         index=df_trans.index, columns=df_trans.columns)


### Store to my_dataset for easy export below.
#my_dataset = data_dict

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)



### Task 4 and 5: Try and tune a variety of classifiers with Pipeline and GridSearchCV

# import sklearn classifiers
from sklearn.svm import SVC
svc = SVC()

"""
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()

# Create a procedue to take feature list and result from pipeline grid search
# and return cross-validation evalutating metrics using tester.py module
def performance(old_list, grid_result):
    selector = gird_result.named_steps['selector']
    k_features = gird_result.named_steps['selector'].get_params(deep=True)['k']
    print "Number of features selected: %i" %(k_features)
    selected = selector.fit_transform(df_scaled[old_list], df_scaled['poi'])
    scores = zip(old_list, selector.scores_, selector.pvalues_)
    sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
    new_list = list(map(lambda x: x[0], sorted_scores))[0:k_features]
    new_list = ['poi']+ new_list
    new_dataset = df_scaled[new_list].to_dict(orient = 'index')  
    new_clf = gird_result.named_steps['clf']
    tester.dump_classifier_and_data(new_clf, new_dataset, new_list)
    tester.main()
    print "\nThis took %.2f seconds\n" %(time() - start)
    print "--------------------------------------------------------"

# Build pipeline with selector and clf steps
# and iterate 3 classifiers (svc, gnb, and neigh) with their parameter sets
# and iterate 3 feature lists:  1. features_list, 2. total_features, 3. rel_total_features
from sklearn.pipeline import Pipeline

# Declare paremeters grid
parameters = {svc: {'selector__k':[19, 15, 10, 7], \
                     'clf__kernel': ['rbf', 'linear', 'poly'], \
                     'clf__C': [0.1, 1, 10, 100, 1000], \
                     'clf__gamma': [1, 0.1, 0.01, 0.001, 0.0001], \
                     'clf__class_weight': ['balanced', None]}, \
              gnb: {'selector__k':[19, 15, 10, 7]}, \
              neigh: {'selector__k':[19, 15, 10, 7], \
                      'clf__n_neighbors': [5, 8, 10, 15], \
                      'clf__weights' : ['uniform','distance'], \
                      'clf__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'], \
                      'clf__metric' : ['euclidean', 'manhattan', 'minkowski']}}

num = 1
for features in [features_list, total_features, rel_total_features]:
    print num
    for classifier in parameters:
        approach1 = Pipeline([('selector', SelectKBest()), \
                      ('clf', classifier)])
        grid_search = GridSearchCV(approach1, parameters[classifier], scoring='f1')
        start = time()
        gird_result = grid_search.fit(df_scaled[features], df_scaled['poi']).best_estimator_
        performance(features, gird_result)
    print "========================================================"
    num += 1
   
# Create a procedue to take feature list and result from pipeline grid search
# and return cross-validation evalutating metrics using tester.py module
def performance_w_pca(old_list, grid_result):
    reducer = gird_result.named_steps['reducer']
    n_components = gird_result.named_steps['reducer'].get_params(deep=True)['n_components']
    print "Number of component: %i" %(n_components)
    reduced = pd.DataFrame(reducer.fit_transform(df_scaled[old_list]), index=df_scaled.index)
    new_list = list(reduced.columns)
    new_list = ['poi']+ new_list
    reduced.insert(0, 'poi', df_scaled.poi)
    new_dataset = reduced.to_dict(orient = 'index') 
    new_clf = gird_result.named_steps['clf']
    tester.dump_classifier_and_data(new_clf, new_dataset, new_list)
    tester.main()
    print "\nThis took %.2f seconds\n" %(time() - start)
    print "--------------------------------------------------------"

# Build pipeline with reducer and clf steps
# and iterate 3 classifiers (svc, gnb, and neigh) with their parameter sets
# and iterate 3 feature lists:  1. features_list, 2. total_features, 3. rel_total_features
from sklearn.decomposition import PCA

parameters = {svc: {'reducer__n_components':[1, 2, 3, 5, 7, 10], \
                    'clf__kernel': ['rbf', 'linear', 'poly'], \
                    'clf__C': [0.1, 1, 10, 100, 1000], \
                    'clf__gamma': [1, 0.1, 0.01, 0.001, 0.0001], \
                    'clf__class_weight': ['balanced', None]}, \
              gnb: {'reducer__n_components':[1, 2, 3, 5, 7, 10]}, \
              neigh: {'reducer__n_components':[1, 2, 3, 5, 7, 10], \
                      'clf__n_neighbors': [5, 8, 10, 15], \
                      'clf__weights' : ['uniform','distance'], \
                      'clf__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'], \
                      'clf__metric' : ['euclidean', 'manhattan', 'minkowski']}}

num = 1
for features in [features_list, total_features, rel_total_features]:
    print num
    for classifier in parameters:
        approach2 = Pipeline([('reducer', PCA()), \
                      ('clf', classifier)])
        grid_search = GridSearchCV(approach2, parameters[classifier], scoring='f1')
        start = time()
        gird_result = grid_search.fit(df_scaled[features], df_scaled['poi']).best_estimator_
        performance_w_pca(features, gird_result)
    print "========================================================"
    num += 1    

"""

# Select 7 features that have highest ANOVA F-value with the factor by poi label
# Create finalized_features_list with selected 7 features
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=7)
selected7 = selector.fit_transform(df_scaled[rel_total_features], df_scaled['poi'])
scores = zip(rel_total_features, selector.scores_, selector.pvalues_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
finalized_features_list = list(map(lambda x: x[0], sorted_scores))[0:7]

# Finalize features list, dataset, and classifier based on pipeline gridsearch results
features_list = ['poi']+ finalized_features_list
my_dataset = df_scaled[features_list].to_dict(orient = 'index')  
clf = SVC(C=100, cache_size=200, class_weight='balanced', coef0=0.0, \
              decision_function_shape=None, degree=3, gamma=1, kernel='poly', \
              max_iter=-1, probability=False, random_state=None, \
              shrinking=True, tol=0.001, verbose=False)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)