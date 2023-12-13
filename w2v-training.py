import numpy as np
from gensim.models import Word2Vec, FastText

# Read in the training data and split into sentences

counter = 0
sent = []
sentences = []
with open('data/conll2003.train.conll', 'r', encoding='utf8') as infile:
    for line in infile.readlines():
        comps = line.split()
        if len(comps) > 0:
            counter += 1 
            sent.append(comps[0])
        else:
            counter = 0
            sentences.append(sent)
            sent = []

# Import and fine-tune the Google News word2vec model on the training data
"""
w2vModel = Word2Vec(vector_size=300, min_count=1, window=10)
w2vModel.build_vocab(sentences)
w2vModel.wv.vectors_lockf = np.ones(len(w2vModel.wv))
w2vModel.wv.intersect_word2vec_format("C:/Users/yoshm/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz", binary=True)
w2vModel.train(sentences, total_examples=len(sentences), epochs=10)
w2vModel.save("models/w2vtuned.model")
"""

# Train a FastText model on the training data
FTmodel = FastText(sentences=sentences, vector_size=300, window=10, min_count=1)
FTmodel.save("models/FT.model")