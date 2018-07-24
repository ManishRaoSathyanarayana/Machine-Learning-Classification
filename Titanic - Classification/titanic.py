#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 07:19:01 2018

@author: surbhikhandelwal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Import data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])

data.info()

# Impute missing numerical variables
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())
data['Embarked'].fillna('S', inplace=True)
replacement = {
    'S': 0,
    'Q': 1,
    'C': 2
}
data['Embarked'] = data['Embarked'].apply(lambda x: replacement.get(x))
data['Embarked'] = StandardScaler().fit_transform(data['Embarked'].values.reshape(-1, 1))
data['Cabin'].fillna('U', inplace=True)
data['Cabin'] = data['Cabin'].apply(lambda x: x[0])
data['Cabin'].unique()

replacement = {
    'T': 0,
    'U': 1,
    'A': 2,
    'G': 3,
    'C': 4,
    'F': 5,
    'B': 6,
    'E': 7,
    'D': 8
}

data['Cabin'] = data['Cabin'].apply(lambda x: replacement.get(x))
data['Cabin'] = StandardScaler().fit_transform(data['Cabin'].values.reshape(-1, 1))
# Check out info of data
data.info()

data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head()


# Select columns and view head
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp', 'Parch', 'Cabin', 'Embarked']]
data.head()

data_train = data.iloc[:891]
data_test = data.iloc[891:]

X = data_train.values
test = data_test.values
y = survived_train.values


# Instantiate model and fit to data
clf = DecisionTreeClassifier(criterion='entropy', max_depth=6)
clf.fit(X, y)

 #Make predictions and store in 'Survived' column of df_test
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('predictions/decision_tree.csv', index=False)

#KAGGLE ACCURACY : 77.99%

#Applying K Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X =X, y = y, cv = 10)
accuracies.mean()
accuracies.std()

#Applying Grid search - find best model & best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'max_depth':[5,6,7,8,9,10], 'criterion': ['gini']},
               {'max_depth':[5,6,7,8,9,10], 'criterion': ['entropy']}]
grid_search = GridSearchCV(estimator = clf, param_grid = parameters, 
                          scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X, y)
best_accuracy = grid_search.best_score_
best_paramaters =grid_search.best_params_




#Support Vector Machine Classifier

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(C =1000, kernel = 'linear', random_state = 0)
classifier.fit(X, y)

# Predicting the Test set results
y_prediction = classifier.predict(test)
df_test['Survived'] = y_prediction

df_test[['PassengerId', 'Survived']].to_csv('predictions/SVM.csv', index=False)

#KAGGLE ACCURACY 75%

#Applying K Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X =X, y = y, cv = 10)
accuracies.mean()
accuracies.std()

#Applying Grid search - find best model & best parameters
from sklearn.model_selection import GridSearchCV
#specify different parameters
#parameters = [{'C':[1, 10, 100, 1000], 'kernel': ['linear']},
 #              {'C':[1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001]}]
# change gamma to find more accurate value
parameters = [{'C':[1, 10, 100, 1000, 10000], 'kernel': ['linear']},
               {'C':[1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, 
                          scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X, y)
best_accuracy = grid_search.best_score_
best_paramaters =grid_search.best_params_

#Naive Bayes
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(test)
df_test['Survived'] = y_pred
df_test[['PassengerId', 'Survived']].to_csv('predictions/Naive Bayes.csv', index=False)
#Kaggle Accuracy: 74.6 %



#KNN
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 1)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(test)
df_test['Survived'] = y_pred
df_test[['PassengerId', 'Survived']].to_csv('predictions/KNN.csv', index=False)


from sklearn.model_selection import GridSearchCV
parameters = [{'n_neighbors':[1, 2, 3, 4, 5,6,7,8,9,10], 'metric': ['euclidean'], 'p' :[1,2,3,4,5,6]},
               {'n_neighbors':[1, 2, 3, 4, 5,6,7,8,9,10], 'metric': ['minkowski'], 'p' :[1,2,3,4,5,6]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, 
                          scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X, y)
best_accuracy = grid_search.best_score_
best_paramaters =grid_search.best_params_
#Kaggle Accuracy 67%
 



#ANN

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
test = sc.transform(test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 8))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, y, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(test)
y_pred = (y_pred > 0.5)
y_pred = y_pred.astype(int)
df_test['Survived'] = y_pred
df_test[['PassengerId', 'Survived']].to_csv('predictions/ANN.csv', index=False)

