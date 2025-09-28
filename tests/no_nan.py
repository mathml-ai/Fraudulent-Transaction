import numpy as np
import pandas as pd
import joblib

def test_no_nan_for_non_tree_models(load_sample_data):
    model = joblib.load("fraud_model.pkl")
    X = load_sample_data.X

    # Check model type (example: sklearn LogisticRegression or SVC)
    non_tree_models = (
    "LogisticRegression",
    "SVC",                # Support Vector Classifier
    "LinearSVC",          # Linear Support Vector Classifier
    "KNeighborsClassifier",
    "GaussianNB",         # Naive Bayes
    "MultinomialNB",
    "BernoulliNB",
    "ComplementNB",
    "RidgeClassifier",
    "RidgeClassifierCV",
    "SGDClassifier",      # Linear models with stochastic gradient descent
    "Perceptron",
    "PassiveAggressiveClassifier",
    "QuadraticDiscriminantAnalysis",
    "LinearDiscriminantAnalysis",
)

    if any(name in str(type(model)) for name in non_tree_models):
        assert not np.isnan(X.values).any(), "NaN values present in input data!"
