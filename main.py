from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer

import pandas as pd
import numpy as np
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from gensim.models import Word2Vec, FastText
import gensim.downloader as api


class NER:

    def __init__(self, modelname="", trainingfile="data/conll2003.train.conll", inputfile="data/conll2003.test.conll"):
        self.trainingfile = trainingfile
        self.inputfile = inputfile
        self.outputfile = "out/" + modelname + "tuned.txt"
        self.modelname = modelname
        self.modelout = "models/" + modelname + "tuned.pkl"
        self.lem = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

    
    def convert_tag(self, nltk_tag):
        """Converts the NLTK POS tags to WordNet tags."""
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None


    def extract_features_and_labels(self, filename, simple=False):
        
        tokens = []

        with open(filename, 'r', encoding='utf8') as infile:
            for line in infile:
                components = line.rstrip('\n').split()
                if len(components) > 0:
                    token = components[0]
                    tokens.append([token, components[1]])
        
        wordset = set([token for token, _ in tokens])
        features = []
        ne = []

        if not simple:
            # Load the word embeddings
            # wv = api.load('word2vec-google-news-300')
            wv = Word2Vec.load("models/w2vtuned.model").wv
            # wv = FastText.load("models/FT.model").wv

        with open(filename, 'r', encoding='utf8') as infile:
            for i, line in enumerate(infile):
                components = line.rstrip('\n').split()
                if len(components) > 0:

                    # gold is in the last column
                    ne.append(components[-1])

                    if simple:
                        features.append({'token': components[0]})
                    
                    else:
                        token = components[0]
                        feature_dict = {'pos': components[1], 'chunk': components[2]}
                        
                        # lemmatize the token
                        if self.convert_tag(components[1]) is not None:
                            feature_dict["lemma"] = self.lem.lemmatize(token, self.convert_tag(components[1]))
                        else:
                            feature_dict["lemma"] = self.lem.lemmatize(token)

                        # add info about previous and next tokens
                        feature_dict["prev_token"] = tokens[i-1][0] if 0 < i < len(tokens) else ""
                        feature_dict["prev_pos"] = tokens[i-1][1] if 0 < i < len(tokens) else ""
                        feature_dict["next_token"] = tokens[i+1][0] if 0 < i < len(tokens)-1 else ""
                        feature_dict["next_pos"] = tokens[i+1][1] if 0 < i < len(tokens)-1 else ""
                        
                        # check if the token is acronym
                        if token.isupper() and len(token) <= 5 and token.lower() not in self.stopwords:
                            feature_dict['is_acronym'] = True
                        else:
                            feature_dict['is_acronym'] = False
                        
                        # check if the token starts with a capital letter
                        if token[0].isupper() and token[1:].islower():
                            feature_dict['is_cap'] = True
                        else:
                            feature_dict['is_cap'] = False
                        
                        # check if the token exists in the training data
                        if token not in wordset:
                            feature_dict['is_oov'] = True
                        else:
                            feature_dict['is_oov'] = False
                            
                        # add word embeddings
                        for j in range(300):
                            if token in wv:
                                feature_dict['wv' + str(j)] = wv[token][j]
                            else:
                                feature_dict['wv' + str(j)] = 0
                        
                        features.append(feature_dict)
                    
        return features, ne


    def create_classifier(self, train_features, train_targets):
        """Creates a classifier based on the training data."""
        
        model = LogisticRegression(solver='lbfgs', max_iter=2000)
            
        if self.modelname == "nb":
            model = GaussianNB()
            
        if self.modelname == "svm":
            model = SVC(kernel='linear', C=1, tol=0.01, gamma=0.1)
        
        # Vectorisation
        vec = DictVectorizer()
        features_vectorized = vec.fit_transform(train_features)
        
        # Fitting the model
        if self.modelname == 'nb':
            fitted = model.fit(features_vectorized.toarray(), train_targets)
        else:
            fitted = model.fit(features_vectorized, train_targets)
            
        with open(self.modelout, 'wb') as f:
            pickle.dump(fitted, f)
            
        # with open("models/vec.pkl", 'wb') as f:
        #   pickle.dump(vec, f)
        
        return fitted, vec
    

    def classify_data(self, model, vec):
        """Classifies data and writes it to an output file."""
        
        # Extract, vectorise, and predict input data
        features = self.extract_features_and_labels(self.inputfile)[0]
        features_vectorized = vec.transform(features)
        
        if self.modelname == 'nb':
            predictions = model.predict(features_vectorized.toarray())
        else:
            predictions = model.predict(features_vectorized)
        
        # Write the predictions to the output file
        with open(self.outputfile, 'w') as outfile:
            counter = 0
            for line in open(self.inputfile, 'r'):
                if len(line.rstrip('\n').split()) > 0:
                    outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
                    counter += 1


    def train(self):

        training_features, gold_labels = self.extract_features_and_labels(self.trainingfile)
        print("Extracted features and labels.")
        ml_model, vectoriser = self.create_classifier(training_features, gold_labels)
        print("Created classifier.")
        self.classify_data(ml_model, vectoriser)
        print("Classified data.")


if __name__ == '__main__':

    name = "svm"
    tfile = "data/conll2003.train.conll"
    ifile = "data/conll2003.test.conll"

    print(f'Training {name} model.')

    ner = NER(name)

    ner.train()
