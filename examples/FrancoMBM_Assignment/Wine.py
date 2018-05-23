
# importing libraries
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import IPython
from IPython.display import display
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# importing datasets

raw_df_red = pd.read_csv(r"C:\Users\franc\Desktop\SecondDesk\DataScienceCertificate\Classes\Assingments\MachineLearning\Homework\winequality-red.csv", sep =';')
raw_df_white = pd.read_csv(r"C:\Users\franc\Desktop\SecondDesk\DataScienceCertificate\Classes\Assingments\MachineLearning\Homework\winequality-white.csv", sep =';')

# exploring datasets

raw_df_red.describe()
raw_df_white.describe()
raw_df_white.info()

#-------------------------------white whine selection--------------------------

X = raw_df_white.iloc[:,:-1].values # independent variables X
y = raw_df_white['quality'].values # dependent Variables y

X_train_white, X_test_white, y_train_white, y_test_white = cross_validation.train_test_split(X, y, test_size = 0.2, random_state = 0)

# visual data exploration
X_train = raw_df_white.iloc[:,:-1]
y_train = raw_df_white['quality']
pd.plotting.scatter_matrix(X_train, c = y_train, figsize = (30, 30), marker ='o', hist_kwds = {'bins': 20},
                            s = 60, alpha = 0.7)

#before scaling
plt.boxplot(X_train_white, manage_xticks = False)
plt.yscale("symlog")
plt.xlabel("Features")
plt.ylabel("Target Variable")
plt.show()

scaler = StandardScaler()
#scaler =  MinMaxScaler()
#scaler = Normalizer()
X_train_white = scaler.fit(X_train_white).transform(X_train_white)
X_test_white = scaler.fit(X_test_white).transform(X_test_white)
# after scaling
plt.boxplot(X_train_white, manage_xticks = False)
plt.yscale("symlog")
plt.xlabel("Features")
plt.ylabel("Target Variable")
plt.show()

# performing PCA

from sklearn.decomposition import PCA
pca = PCA(n_components = None) # input a number for feature extraction

X_train_white = pca.fit_transform(X_train_white)
X_test_white = pca.transform(X_test_white)
explained_var = pca.explained_variance_ratio_
print (explained_var)

#-----------------KNN--------------------------------------

knn = KNeighborsClassifier(n_neighbors = 10, metric = 'manhattan', weights = 'distance', algorithm = 'auto')
knn.fit(X_train_white, y_train_white)
predicted_knn = knn.predict(X_test_white)
# print("Predictions: {}".format(predicted_knn))

scores = cross_val_score(knn, X = X_train_white, y = y_train_white)
print ("Cross Validation Scores: {}".format(scores))

report = classification_report(y_test_white, predicted_knn)
print (report)

# Finding the best parameters for knn:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

params2 = [{'n_neighbors': [1,10,50,100], 'algorithm': ['auto','ball_tree','kd_tree' ],
            'weights': ['uniform', 'distance'], 'metric': ['minkowski', 'manhattan']}]

grid_search = GridSearchCV(estimator = knn, param_grid = params2, scoring = 'accuracy', cv = 5, n_jobs = 1)
grid_search = grid_search.fit(X_train_white, y_train_white)
accuracy = grid_search.best_score_
best_params = grid_search.best_params_
print(accuracy)
print(best_params)

train_accuracy = []
test_accuracy = []

neighbors = range(1,100,10)
algorithms = ['auto', 'ball_tree', 'kd_tree']
weights = ['uniform', 'distance']

for i in neighbors:
    knn = KNeighborsClassifier(n_neighbors = i, metric = 'manhattan', weights = 'distance', algorithm = 'auto')
    knn.fit(X_train_white, y_train_white)
    train_accuracy.append(knn.score(X_train_white, y_train_white))
    test_accuracy.append(knn.score(X_test_white, y_test_white))
plt.plot(neighbors, train_accuracy, label = 'Train set accuracy')
plt.plot(neighbors, test_accuracy, label = 'Test set accuracy')
plt.ylabel("Accuracy")
plt.xlabel("Number of neighbors")
plt.legend()
plt.show()

#------------------------------- Kernel SVC:----------------------------------
from sklearn.svm import SVC
svm = SVC(C = 1000, kernel = 'rbf', gamma = 1)
svm.fit(X_train_white, y_train_white)
predicted = svm.predict(X_test_white)
#print("Predictions: {}".format(predicted))scores = cross_val_score(svm, X = X_train_white, y = y_train_white)
report = classification_report(y_test_white, predicted)
print (report)
# print ("Cross Validation Scores: {}".format(scores))

# -----------Finding the best parameters for SVC----------

params = [{'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 0.001]}]

grid_search = GridSearchCV(estimator = svm, param_grid = params, scoring = 'accuracy', cv = 5, n_jobs =1)
grid_search = grid_search.fit(X_train_white, y_train_white)
accuracySVC = grid_search.best_score_
best_paramsSVC = grid_search.best_params_

print (accuracySVC)
print (best_paramsSVC)




train_accuracy = []
test_accuracy = []

Ci = [10, 100, 1000]

for i in Ci:
    svm = SVC(C = i, kernel = 'rbf', gamma = 1) # try rbf, linear and poly
    svm.fit(X_train_white, y_train_white)
    train_accuracy.append(svm.score(X_train_white, y_train_white))
    test_accuracy.append(svm.score(X_test_white, y_test_white))
plt.plot(Ci, train_accuracy, label = 'Train set accuracy')
plt.plot(Ci, test_accuracy, label = 'Test set accuracy')
plt.ylabel("Accuracy")
plt.xlabel("C")
plt.legend()
plt.show()


####---------XGBoost-----------------


from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor

xclas = XGBClassifier()  # for classifier
xclas.fit(X_train_white, y_train_white)
y_pred = xclas.predict(X_test_white)

cross_val_score(xclas, X_train_white, y_train_white)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_white, y_pred)
print (cm)
