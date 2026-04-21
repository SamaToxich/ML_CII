import math
import sys
from collections import Counter
import numpy as np
import pandas as pd

np.random.seed(1)

alpha = 0.05
iterations = 2
hidden_size = 50
window = 2
negative = 5
wordcnt = Counter()
word2index = {}
input_dataset =list()
concatenated = list()

target = np.zeros(negative+1)
target[0] = 1

df = pd.read_csv('IMDB Dataset.csv')
raw_reviews = df['review'].tolist()[0:5000]

tokens = list(map(lambda x:(x.split()), raw_reviews))
wordcnt = Counter()

for sent in tokens:
    for word in sent:
        wordcnt[word] += 1
vocab = list(set(map(lambda x:x[0],wordcnt.most_common())))

word2index = {word: i for i, word in enumerate(vocab)}

for sent in tokens:
    sent_indices = list()
    for word in sent:
        try:
            sent_indices.append(word2index[word])
            concatenated.append(word2index[word])
        except:
            ""
    input_dataset.append(sent_indices)
concatenated = np.array(concatenated)

weights_1 = (np.random.rand(len(vocab), hidden_size) - 0.5) * 0.2
weights_2 = np.random.rand(len(vocab), hidden_size) * 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def similar(target_word='beautiful'):

    target_i = word2index[target_word]
    scores = Counter()

    for w, i in word2index.items():
        raw_difference = weights_1[i] - (weights_1[target_i])
        squared_difference = raw_difference * raw_difference
        scores[w] = -math.sqrt(sum(squared_difference))

    return scores.most_common(10)

for i, review in enumerate(input_dataset * iterations):

    for target_i in range(len(review)):

        target_samples = [review[target_i]] + concatenated[np.random.randint(0, len(concatenated), negative)].tolist()

        left_context = review[max(0, target_i - window): target_i]
        right_context = review[target_i+1: min(len(review), target_i + window)]

        layer_1 = np.mean(weights_1[left_context+right_context], axis=0)
        layer_2 = sigmoid(layer_1 @ weights_2[target_samples].T)

        layer_2_delta = layer_2 - target
        layer_1_delta = layer_2_delta @ weights_2[target_samples]

        weights_2[target_samples] -= np.outer(layer_2_delta, layer_1) * alpha
        weights_1[left_context+right_context] -= layer_1_delta * alpha

    if i % 250 == 0:
        sys.stdout.write('\rProgress:'+str(i/float(len(input_dataset)*iterations)) + " " + str(similar('terrible')))
    sys.stdout.write('\rProgress:'+str(i/float(len(input_dataset)*iterations)))
print(similar('terrible'))
