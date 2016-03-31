#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
#from class_vis import prettyPicture
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.cross_validation import cross_val_score

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]



#### initial visualization
#plt.xlim(0.0, 1.0)
#plt.ylim(0.0, 1.0)
#plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
#plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
#plt.legend()
#plt.xlabel("bumpiness")
#plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

clf = AdaBoostClassifier(n_estimators=100,learning_rate=1)
clf.fit(features_train, labels_train)
#scores = cross_val_score(clf, features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "accuracy Adaboost = ", acc


clf1 = RandomForestClassifier(n_estimators=100,criterion='gini',min_samples_leaf=5,max_depth=2,random_state=1000)
clf1.fit(features_train, labels_train)
pred1 = clf1.predict(features_test)
acc1 = accuracy_score(pred1, labels_test)
print "accuracy RF = ", acc1

clf2 = GaussianNB()
clf2.fit(features_train, labels_train)
accuracy = clf2.score(features_test, labels_test, sample_weight=None)
print "accuracy NaiveBayes = ", accuracy

clf3 = svm.SVC(kernel='rbf',C=100000)
clf3.fit(features_train, labels_train)
pred3 = clf3.predict(features_test)
acc3 = accuracy_score(pred3, labels_test)
print "accuracy SVM = ", acc3

clf4 = tree.DecisionTreeClassifier(min_samples_split=40)
clf4.fit(features_train, labels_train)
pred4 = clf4.predict(features_test)
acc4 = accuracy_score(pred4, labels_test)
print "accuracy DecisionTrees = ", acc4 #, acc1

#try:
#    prettyPicture(clf, features_test, labels_test)
#except NameError:
#    pass
