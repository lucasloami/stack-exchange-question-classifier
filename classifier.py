#!/usr/bin/python

from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import json
import numpy as np

data = json.load(open('data/training.json'))
df = CountVectorizer(stop_words='english', strip_accents='unicode',
    lowercase=True, min_df=3, max_df=0.9, ngram_range=(1, 2), max_features=19000)
data_test = [d['excerpt'] for d in data]
X = df.fit_transform(data_test).toarray()
Y = np.array([d['topic'] for d in data])
naive_bayes = MultinomialNB()
naive_bayes.fit(X, Y)

test_data = json.load(open('data/input00.txt'))
test_data_test = [d['excerpt'] for d in test_data]
test_X = df.transform(test_data_test).toarray()
test_Y = np.array(open('data/output00.txt').read().splitlines())

success = 0
fail = 0

for x, y in zip(test_X, test_Y):
    h = naive_bayes.predict(x)
    if h == y:
        success += 1
    else:
        fail += 1

print 'Success: ', success/(success + fail)
print df.vocabulary_