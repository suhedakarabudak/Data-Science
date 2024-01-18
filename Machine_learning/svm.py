# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn import datasets
from sklearn import metrics

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)



#Define support vector classifier with hyperparameters


svc = SVC(random_state=101)

svc.fit(X_train,y_train)
accuracies = cross_val_score(svc,X_train,y_train)

print("Train Score:",np.mean(accuracies))

print("Test Score:",svc.score(X_test,y_test))


grid = {

    'C':[0.01,0.1,1,10],

    'kernel' : ["linear","rbf"],

    'degree' : [1,3,7],

    'gamma' : [0.01,1]

}

svm  = SVC ()

svm_cv = GridSearchCV(svm, grid, cv = 5)

svm_cv.fit(X_train,y_train)

print("Best Parameters:",svm_cv.best_params_)

print("Train Score:",svm_cv.best_score_)