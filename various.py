import numpy as np
from scipy.io import  arff
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
data,meta = arff.loadarff('iris.arff')
X = np.array([list(row)[:-1] for row in data], dtype=np.float32)
y = np.array([row[-1].decode('utf-8') for row in data])

clf1=BaggingClassifier()
scores = cross_val_score(clf1, X, y, cv=5)
print("Bagging Classifier Accuracy: ", scores.mean())
clf2=RandomForestClassifier()
scores = cross_val_score(clf2, X, y, cv=5)
print("Random Forest Classifier Accuracy: ", scores.mean())
clf3=AdaBoostClassifier()
scores = cross_val_score(clf3, X, y, cv=5)
print("AdaBoost Classifier Accuracy: ", scores.mean())
clf4=GradientBoostingClassifier()
scores = cross_val_score(clf4, X, y, cv=5)
print("Gradient Boosting Classifier Accuracy: ", scores.mean())