# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:18:49 2018

@author: Michael
"""

#  Lab 2 - Lo
import os

basepath = '.'

outpath = os.path.join (basepath, "out")

if not os.path.exists(outpath):
    os.makedirs(outpath)

import pandas as pd

dataset = pd.read_csv(os.path.join (basepath, 'Data.csv'))

import numpy as np

seed = 7
np.random.seed(seed)
# Lab 3 - Data Processing and finding data correlations

#Encode columns using label encoding 
from sklearn.preprocessing import LabelEncoder

responsetimeencoder = LabelEncoder()
dataset['Projected ROI'] = responsetimeencoder.fit_transform(dataset['Projected ROI'])

suppliesgroupencoder = LabelEncoder()
dataset['Team Size'] = suppliesgroupencoder.fit_transform(dataset['Team Size'])

suppliessubgroupencoder = LabelEncoder()
dataset['Project Cost'] = suppliessubgroupencoder.fit_transform(dataset['Project Cost'])

regionencoder = LabelEncoder()
dataset['Actual ROI'] = regionencoder.fit_transform(dataset['Actual ROI'])


##What are the correlations between columns and target
correlations = dataset.corr()['Team Size'].sort_values()

#routetomarketencoder = LabelEncoder()
#dataset['Route To Market'] = routetomarketencoder.fit_transform(dataset['Route To Market'])
#correlations = dataset.corr()['Opportunity Result'].sort_values()

# Lab 4 - Encoding and scaling the data
#Throw out unneeded columns 
dataset = dataset.drop('Project Name', axis=1)

#One Hot Encode columns that are more than binary
# avoid the dummy variable trap
dataset = pd.concat([pd.get_dummies(dataset['Projected ROI'], prefix='Projected ROI', drop_first=True),dataset], axis=1)
dataset = dataset.drop('Projected ROI', axis=1)

##What are the correlations between columns and target
dataset = pd.concat([pd.get_dummies(dataset['Team Size'], prefix='Team Size', drop_first=True),dataset], axis=1)
dataset = dataset.drop('Team Size', axis=1)


dataset = pd.concat([pd.get_dummies(dataset['Project Cost'], prefix='Project Cost', drop_first=True),dataset], axis=1)
dataset = dataset.drop('Project Cost', axis=1)

#dataset = pd.concat([pd.get_dummies(dataset['Actual ROI'], prefix='Actual ROI', drop_first=True),dataset], axis=1)
#dataset = dataset.drop('Actual ROI', axis=1)

#Create the input data set (X) and the outcome 󰀀
X = dataset.drop('Actual ROI', axis=1).iloc[:, 0:dataset.shape[1] - 1].values
y = dataset.iloc[:, dataset.columns.get_loc('Actual ROI')].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# Lab 5 - Creationg the ANN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
# Initilzing the ANN
model = Sequential()

#Adding the input layer
model.add(Dense(units = 8, activation = 'relu', input_dim=X.shape[1], name= 'Input_Layer'))
#Add hidden layer
model.add(Dense(units = 8, activation = 'relu', name= 'Hidden_Layer_1'))
#Add the output layer
model.add(Dense(units = 1, activation = 'sigmoid', name= 'Output_Layer'))
# compiling the ANN
model.compile(optimizer= 'nadam', loss = 'binary_crossentropy', metrics=['accuracy'])
# Lab 6 - Model training and Visualization
print(model.summary())

#Fit the ANN to the training set
history = model.fit(X, y, validation_split = .20, batch_size = 64, epochs = 200)

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Lab 7 - Improve your model

##Evaluating the ANN
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Dense
#
#def build_classifier(optimizer):
#   model = Sequential()
#   model.add(Dense(units = 24, activation = 'relu', input_dim=X.shape[1], name= 'Input_Layer'))
#   model.add(Dense(units = 24, activation = 'relu', name= 'Hidden_Layer_1'))
#   model.add(Dense(1, activation = 'sigmoid', name= 'Output_Layer'))
#   model.compile(optimizer= optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
#   return model
#
#parameters = {'batch_size': [64],
#              'epochs': [25],
#              'optimizer': ['adam','sgd','adamax','nadam']}
#
#grid_search = GridSearchCV(estimator = classifier,
#   param_grid = parameters,
#   scoring = 'accuracy',
#   verbose = 5)
#grid_search = grid_search.fit(X, y)
#
#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_
##Lab 8 - Save your model