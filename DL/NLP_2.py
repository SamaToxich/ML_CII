import math
import re
import sys
from collections import Counter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize

np.random.seed(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def clean_text_nltk(text):
    # Удаляем HTML
    text = re.sub(r'<[^>]+>', ' ', text)
    # Токенизация
    tokens = word_tokenize(text.lower())
    # Оставляем только буквенные токены длиной > 1
    tokens = [t for t in tokens if t.isalpha() and len(t) > 1]
    return tokens

def similar(target_word='beautiful'):

    target_i = word2index[target_word]
    scores = Counter()

    for w, i in word2index.items():
        raw_difference = weights_1[i] - (weights_1[target_i])
        squared_difference = raw_difference * raw_difference
        scores[w] = -math.sqrt(sum(squared_difference))

    return scores.most_common(10)

def analogy(positive=['terrible', 'good'], negative=['bad']):
    norms = np.sqrt((np.sum(weights_1 ** 2,axis=1, keepdims=True)))

    normed_weights = weights_1 * norms

    query_vect = np.zeros(len(weights_1[0]))

    query_vect += np.sum([normed_weights[word2index[word]] for word in positive], axis=0)
    query_vect -= np.sum([normed_weights[word2index[word]] for word in negative], axis=0)

    scores = Counter()
    for word, index in word2index.items():
        raw_diff = weights_1[index] - query_vect
        Wdiff = raw_diff * raw_diff

        scores[word] = -math.sqrt(sum(Wdiff))

    # --- ВИЗУАЛИЗАЦИЯ ---
    # Берём первые 2 компоненты весов (PCA-like упрощение)
    xs = [weights_1[word2index[w]][0] for w in positive + negative]
    ys = [weights_1[word2index[w]][1] for w in positive + negative]

    plt.figure(figsize=(6,6))
    plt.scatter(xs[:len(positive)], ys[:len(positive)], c='green', label='positive', s=100)
    plt.scatter(xs[len(positive):], ys[len(positive):], c='red', label='negative', s=100)

    # Подписи
    for i, w in enumerate(positive + negative):
        plt.annotate(w, (xs[i], ys[i]), xytext=(5,5), textcoords='offset points')

    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Words (first 2 dims)')
    plt.show()
    # --- КОНЕЦ ВИЗУАЛИЗАЦИИ ---

    return scores.most_common(10)[1:]

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

df = pd.read_csv('../data/IMDB Dataset.csv')
raw_reviews = df['review'].tolist()[0:5000]

tokens = [clean_text_nltk(review) for review in raw_reviews]

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

for epochs, review in enumerate(input_dataset * iterations):

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

    sys.stdout.write(f'\rОбработано: {epochs}/{len(tokens)}')
print(similar('terrible'))
