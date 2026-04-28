import sys,random,math
from collections import Counter
import numpy as np
import re
from nltk.tokenize import word_tokenize

np.random.seed(1)

def softmax(x_):
    x = np.atleast_2d(x_)
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

def clean_text_nltk(text):
    text = re.sub(r'<[^>]+>', ' ', text)

    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and len(word) > 1]
    return tokens

def words2indices(sentence):
    idx = list()
    for word in sentence:
        idx.append(word2index[word])
    return idx

f = open('../data/qa1_single-supporting-fact_train.txt','r')
raw = f.readlines()
f.close()

tokens = [clean_text_nltk(review) for review in raw[0:1000]]

vocab = set()

for sent in tokens:
    for word in sent:
        vocab.add(word)

vocab = list(vocab)

word2index = {word: i for i, word in enumerate(vocab)}

embed_size = 10

weights_1 = (np.random.rand(len(vocab), embed_size) - 0.5) * 0.1 # embed
weights_2 = (np.random.rand(embed_size, len(vocab)) - 0.5) * 0.1 # decoder

recurrent = np.eye(embed_size) # матрица для учета порядка слов
start = np.zeros((1, embed_size))

one_hot = np.eye(len(vocab))

def predict(sent):
    layers = list()
    layer = {}
    layer['hidden'] = start
    layers.append(layer)

    loss = 0

    preds = list()
    for target_i in range(len(sent)):

        layer = {}
        layer['pred'] = softmax(layers[-1]['hidden'] @ weights_2)

        loss += -np.log(layer['pred'][0][sent[target_i]])

        layer['hidden'] = layers[-1]['hidden'] @ recurrent + weights_1[sent[target_i]]
        layers.append(layer)
    return layers, loss

for iter in range(30000):
    alpha = 0.001
    sent = words2indices(tokens[iter%len(tokens)][1:])
    layers, loss = predict(sent)

    for i in reversed(range(len(layers))):
        layer = layers[i]
        target = sent[i-1]

        if i > 0:
            layer['output_delta'] = layer['pred'][0] - one_hot[target]
            new_hidden_delta = layer['output_delta'] @ weights_2.T

            if i == len(layers)-1:
                layer['hidden_delta'] = new_hidden_delta
            else:
                layer['hidden_delta'] = new_hidden_delta + layers[i+1]['hidden_delta'] @ recurrent.T
        else:
            layer['hidden_delta'] = layers[i+1]['hidden_delta'] @ recurrent.T

    start -= layers[0]['hidden_delta'] * alpha / float(len(sent))

    for layers_i, layer in enumerate(layers[1:]):
        weights_2 -= np.outer(layers[layers_i]['hidden'], layer['output_delta']) * alpha / float(len(sent))

        embed_i = sent[layers_i]
        weights_1[embed_i] -= layers[layers_i]['hidden_delta'] * alpha / float(len(sent))

        recurrent -= np.outer(layers[layers_i]['hidden'], layer['hidden_delta']) * alpha / float(len(sent))

    if(iter % 1000 == 0):
        print("Perplexity:" + str(np.exp(loss/len(sent))))

sent_index = 4
l,_ = predict(words2indices(tokens[sent_index]))

print(tokens[sent_index])

for i,each_layer in enumerate(l[1:-1]):
    input = tokens[sent_index][i]
    true = tokens[sent_index][i+1]
    pred = vocab[each_layer['pred'].argmax()]

    print("Prev Input:" + input + (' ' * (12 - len(input))) + \
          "True:" + true + (" " * (15 - len(true))) + "Pred:" + pred)