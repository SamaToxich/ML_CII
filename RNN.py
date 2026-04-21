import sys
import numpy as np
import pandas as pd
from collections import Counter
import math

df = pd.read_csv('IMDB Dataset.csv')
df = df.values.tolist()

raw_reviews = list()
raw_labels = list()
for i in range(len(df)):
    raw_reviews.append(df[i][0])
    raw_labels.append(df[i][1])

tokens = list(map(lambda x:set(x.split(" ")),raw_reviews))

vocab = set()
for sent in tokens:
    for word in sent:
        if len(word) > 0:
            vocab.add(word)
vocab = list(vocab)

word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i

input_dataset = list()
for sent in tokens:
    sent_indices = list()
    for word in sent:
        try:
            sent_indices.append(word2index[word])
        except:
            ""
    input_dataset.append(list(set(sent_indices)))

target_dataset = list()
for label in raw_labels:
    if label == 'positive':
        target_dataset.append(1)
    else:
        target_dataset.append(0)

np.random.seed(1)

alpha = 0.02
iteration = 1
hidden_size = 100

weights_1 = 0.2 * np.random.random((len(vocab), hidden_size)) - 0.1
weights_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1
correct, total = 0, 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def similar(target='beautiful'):
    target_i = word2index[target]
    scores = Counter()
    for w, i in word2index.items():
        raw_difference = weights_1[i] - (weights_1[target_i])
        squared_difference = raw_difference * raw_difference
        scores[w] = -math.sqrt(sum(squared_difference))

    return scores.most_common(10)

for iter in range(iteration):

    for i in range(len(input_dataset) - 1000):

        x, y = input_dataset[i], target_dataset[i]

        layer_1 = sigmoid(np.sum(weights_1[x], axis=0))

        layer_2 = sigmoid(layer_1 @ weights_2)

        layer_2_delta = layer_2 - y
        layer_1_delta = layer_2_delta @ weights_2.T

        weights_1[x] -= layer_1_delta * alpha
        weights_2 -= np.outer(layer_1,layer_2_delta) * alpha

        if np.abs(layer_2_delta) < 0.5:
            correct += 1
        total += 1

        if(i % 10 == 9):
            progress = str(i/float(len(input_dataset)-1000))
            sys.stdout.write('\rlter:'+str(iter+1) \
                             +' Progress:'+progress[2:4] \
                             +'.'+progress[4] \
                             + '% Training Accuracy:' \
                             + str(round(correct/float(total)*100)) + '%')
    print()

test_correct, test_total = 0, 0
for i in range(len(input_dataset)-1000,len(input_dataset)):

    l = input_dataset[i]
    target = target_dataset[i]

    layer_1 = sigmoid(np.sum(weights_1[l],axis=0))
    layer_2 = sigmoid(layer_1 @ weights_2)

    if np.abs(layer_2 - target) < 0.5:
        test_correct += 1
    test_total += 1

print("Test Accuracy:" + str(test_correct / float(test_total)))
