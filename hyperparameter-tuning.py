from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import pandas as pd
import pickle

from main import NER


vec = pickle.load(open("models/vec.pkl", 'rb'))
ner = NER()


def tune_model(vec, inputfile):
    features, targets = ner.extract_features_and_labels(inputfile)
    
    fdf = pd.DataFrame(features)
    fdf["target"] = targets
    samp = fdf.sample(frac=0.2)
    
    vec_features = vec.transform(samp.drop("target", axis=1).to_dict('records'))
    
    gen_params = {'C': [1, 10, 100, 1000], 'tol': [0.01, 0.001, 0.0001], 'gamma': [0.1,1,10]}
    param_grid = [
        {'kernel': ['linear']} | gen_params,
        {'kernel': ['poly'], 'degree': [2, 3, 4]} | gen_params,
        {'kernel': ['rbf']} | gen_params
    ]
    
    svm = SVC()
    clf = GridSearchCV(svm, param_grid, scoring='f1_macro', verbose=2)
    clf.fit(vec_features, samp.target)
    print(sorted(clf.cv_results_.keys()))
    return clf.best_estimator_


test = 'data/conll2003.test.conll'
best = tune_model(vec, test)
print(best)