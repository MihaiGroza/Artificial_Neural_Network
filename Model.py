#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('breast-cancer-wisconsin.data')

#Removing Rows with Missing Data
dataset = dataset.replace('?',np.NaN)
dataset = dataset.dropna()

# Separating Independent Variables
X = dataset.iloc[:, 1:10].values

# Separating Dependent Variable
y = dataset.iloc[:, 10].values

#Data labels already encoded

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Econding Output Layer 
labelencoder_y_1 = LabelEncoder()
y= labelencoder_y_1.fit_transform(y)
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

# Creating Input Layer
classifier.add(Dense(units = 5, kernel_initializer = "uniform", activation= "relu",input_dim=9))

# Creating Hidden Layer
classifier.add(Dense(units = 5, kernel_initializer = "uniform", activation= "relu"))

#Creating Output Layer
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation= "sigmoid"))

#compiling the ANN

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])


# Fitting classifier to the Training set

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch= 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# Cross Validating the Model
def build_classifier():

    classifier = Sequential()


    classifier.add(Dense(units = 5, kernel_initializer = "uniform", activation= "relu",input_dim=9))

    classifier.add(Dense(units = 5, kernel_initializer = "uniform", activation= "relu"))

    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation= "sigmoid"))
    
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

























