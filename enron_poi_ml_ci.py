#!/usr/bin/python
'''
Created by Cosimo Inserra
github: https://github.com/cinserra
Twitter: @COSMO_83

Udacity machine-learnign final project on the Enron dataset
'''
#import modules needed
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

from numpy import log, sqrt, float64, nan

from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, make_scorer, f1_score, recall_score, precision_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import BernoulliRBM
from sklearn.cluster import KMeans

### Features lists
features_poi = ['poi']

### Separated lists in order to apply PCA to each one.
### Emails looks for underlying feature of constant communication between POI's
features_email = [
                "from_messages",
                "from_poi_to_this_person",
                "from_this_person_to_poi",
                "shared_receipt_with_poi",
                "to_messages"
                ]
### Financial features, which could have hints of bribe money
features_finance = [
                "bonus",
                "deferral_payments",
                "deferred_income",
                "director_fees",
                "exercised_stock_options",
                "expenses",
                "loan_advances",
                "long_term_incentive",
                "other",
                "restricted_stock",
                "restricted_stock_deferred",
                "salary",
                "total_payments",
                "total_stock_value"
                ]
#### New features tested
features_new = [
                # Email ratio
                "poi_ratio_messages",
                # Logaritmic Features
                "log_total_payments",
                "log_bonus",
                "log_salary",
                "log_total_stock_value",
]

### features_list is a list of strings, each of which is a feature name.
features_list = features_poi + features_email + features_finance + features_new
NAN_value = 'NaN'
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Defining function in order to remove outliers
def preprocess_data_wo():
    '''
    Loads and removes outliers from the dataset. Returns the data as a
    Python dictionary.
    '''
    ### load the dictionary containing the dataset
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

    ### reoving outliers that do not carry useful information
    outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
    for outlier in outliers:
        data_dict.pop(outlier, 0)

    return data_dict

### Adding features
def add_features(data_dict, features_list, financial_log=False, financial_squared=False):
    '''
    Given the data dictionary of people with features, adds some features to
    '''
    for name in data_dict:

        # Add ratio of POI messages to total.
        try:
            total_messages = data_dict[name]['from_messages'] + data_dict[name]['to_messages']
            poi_related_messages = data_dict[name]["from_poi_to_this_person"] +\
                                    data_dict[name]["from_this_person_to_poi"] +\
                                    data_dict[name]["shared_receipt_with_poi"]
            poi_ratio = 1.* poi_related_messages / total_messages
            data_dict[name]['poi_ratio_messages'] = poi_ratio
            data_dict[name]['poi_ratio_messages_squared'] = poi_ratio ** 2
        except:
            data_dict[name]['poi_ratio_messages'] = NAN_value

        # If feature is financial, add another variable with log transformation.
        if financial_log:
            for feat in features_finance:
                try:
                    data_dict[name][feat + '_log'] = Math.log(data_dict[name][feat] + 1)
                except:
                    data_dict[name][feat + '_log'] = NAN_value

        # If feature is financial, add squared features
        if financial_squared:
            for feat in features_finance:
                try:
                    data_dict[name][feat + '_squared'] = Math.square(data_dict[name][feat]+1)
                except:
                    data_dict[name][feat + '_squared'] = NAN_value

    return data_dict

### Store to my_dataset for easy export below.
def transform_pca(clf_list):
    '''
    From classifier list to pipeline list of the same classifiers and PCA.
    '''

    pca = PCA()
    params_pca = {"pca__n_components":[2, 3, 4, 5, 10, 15, 20], "pca__whiten": [False]}

    for j in range(len(clf_list)):

        name = "clf_" + str(j)
        clf, params = clf_list[j]

        # Parameters in GridSearchCV need to have double underscores
        # between specific classifiers.
        new_params = {}
        for key, value in params.iteritems():
            new_params[name + "__" + key] = value

        new_params.update(params_pca)
        clf_list[j] = (Pipeline([("pca", pca), (name, clf)]), new_params)

    return clf_list


def scale_features(features):
    '''
    Scale features using the Min-Max algorithm
    '''

    # scale features via min-max
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    features = scaler.fit_transform(features)

    return features


def varius_classifiers():
    # List of tuples of a classifier and its parameters.
    clf_list = []

    clf_linearsvm = LinearSVC()
    params_linearsvm = {"C": [0.5, 1, 5, 10, 100, 10**10],"tol":[0.1, 0.0000000001],"class_weight":['balanced']}
    clf_list.append( (clf_linearsvm, params_linearsvm) )

    clf_tree = DecisionTreeClassifier()
    params_tree = { "min_samples_split":[2, 5, 10, 20],"criterion": ('gini', 'entropy')}
    clf_list.append( (clf_tree, params_tree) )

    clf_random_tree = RandomForestClassifier()
    params_random_tree = {  "n_estimators":[2, 3, 5],"criterion": ('gini', 'entropy')}
    clf_list.append( (clf_random_tree, params_random_tree) )

    clf_adaboost = AdaBoostClassifier()
    params_adaboost = { "n_estimators":[20, 30, 50, 100]}
    clf_list.append( (clf_adaboost, params_adaboost) )

    clf_knn = KNeighborsClassifier()
    params_knn = {"n_neighbors":[2, 5], "p":[2,3]}
    clf_list.append( (clf_knn, params_knn) )

    clf_log = LogisticRegression()
    params_log = {"C":[0.5, 1, 10, 10**2,10**10, 10**20],"tol":[0.1, 0.00001, 0.0000000001],"class_weight":['balanced']}
    clf_list.append( (clf_log, params_log) )

    clf_lda = LinearDiscriminantAnalysis()
    params_lda = {"n_components":[0, 1, 2, 5, 10]}
    clf_list.append( (clf_lda, params_lda) )

    logistic = LogisticRegression()
    rbm = BernoulliRBM()
    clf_rbm = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    params_rbm = {"logistic__tol":[0.0000000001, 10**-20],"logistic__C":[0.05, 1, 10, 10**2,10**10, 10**20],"logistic__class_weight":['balanced'],"rbm__n_components":[2,3,4]}
    clf_list.append( (clf_rbm, params_rbm) )

    return clf_list



def opt_classifier(clf, params, features_train, labels_train, optimize=True):
    '''
    GridSearchCV to find optimal parameters of the classifier.
    '''

    if optimize:
        scorer = make_scorer(f1_score)
        clf = GridSearchCV(clf, params, scoring=scorer)
        clf = clf.fit(features_train, labels_train)
        clf = clf.best_estimator_
    else:
        clf = clf.fit(features_train, labels_train)

    return clf


def opt_classifier_list(clf_list, features_train, labels_train):

    best_estimators = []
    for clf, params in clf_list:
        clf_optimized = opt_classifier(clf, params, features_train, labels_train)
        best_estimators.append( clf_optimized )

    return best_estimators

def train_unsupervised_classifier(features_train, labels_train, pca_pipeline):

    clf_kmeans = KMeans(n_clusters=2, tol = 0.001)

    if pca_pipeline:
        pca = PCA(n_components=2, whiten=False)

        clf_kmeans = Pipeline([("pca", pca), ("kmeans", clf_kmeans)])

    clf_kmeans.fit( features_train )

    return [clf_kmeans]

def train_clf(features_train, labels_train, pca_pipeline=False):

    clf_supervised = varius_classifiers()

    if pca_pipeline:
        clf_supervised = transform_pca(clf_supervised)

    clf_supervised = opt_classifier_list(clf_supervised, features_train, labels_train)
    clf_unsupervised = train_unsupervised_classifier(features_train, labels_train, pca_pipeline)

    return clf_supervised + clf_unsupervised

#########################################
# Quantitative evaluation on the test set
#########################################

def evaluate_clf(clf, features_test, labels_test):
    pred = clf.predict(features_test)
    f1 = f1_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    precision = precision_score(labels_test, pred)

    return f1, recall, precision

def evaluate_clf_list(clf_list, features_test, labels_test):
    clf_list_scores = []
    for clf in clf_list:
        f1, recall, precision = evaluate_clf(clf, features_test, labels_test)
        clf_list_scores.append( (clf, f1, recall, precision) )

    return clf_list_scores


def evaluation_loop(features, labels, pca_pipeline=False, num_iters=1000, test_size=0.3):
    '''
    Evaluation metrics multiple times. Add additional info
    '''
    from numpy import asarray

    evaluation_matrix = [[] for n in range(9)]
    for i in range(num_iters):

        #### Split data into training and test sets
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size)

        ### Tain all models
        clf_list = train_clf(features_train, labels_train, pca_pipeline)

        for j, clf in enumerate(clf_list):
            score = evaluate_clf(clf, features_test, labels_test)
            evaluation_matrix[j].append(score)

    # Make a copy of the classifications list.
    summary_list = {}
    for j, col in enumerate(evaluation_matrix):
        summary_list[clf_list[j]] = ( sum(asarray(col)) )

    sorted_list = sorted(summary_list.keys() , key = lambda k: summary_list[k][0], reverse=True)

    return sorted_list, summary_list


### Import build_email_features
data_dict = preprocess_data_wo()

### To test financial_log and financial_squared features, first turn them to True, then uncomment them up in features_new
data_dict = add_features(data_dict, features_list, financial_log=True, financial_squared=True)

### store to my_dataset for easy export below
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)

sorted_list, summary_list = evaluation_loop(features, labels, pca_pipeline = True, num_iters=10, test_size=0.3)

print sorted_list
print "#"*50
print summary_list
print "#"*50
clf = sorted_list[0]
score_list = summary_list[clf]
print "The best classifier is: ", clf
print "F1, recall, precision: ", score_list

test_classifier(clf, my_dataset, features_list)

### Dump the classifier, dataset, and features_list
### so anyone can check the results.

dump_classifier_and_data(clf, my_dataset, features_list)

### End of code --> Great experience!!!
### Next iteration try to decrease the functions/defintiions and slim the code
