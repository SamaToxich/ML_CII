import random, sys, codecs
import numpy as np

np.random.seed(12345)

with codecs.open('spam.txt', "r", encoding='utf-8', errors='ignore') as f:
    raw = f.readlines()

vocab = set(["<unk>"])
spam = list()
ham = list()

for row in raw:
    spam.append(set(row[:-2].split(" ")))

    for word in spam[-1]:
        vocab.add(word)

with codecs.open('ham.txt', "r", encoding='utf-8', errors='ignore') as f:
    raw = f.readlines()

for row in raw:
    ham.append(set(row[:-2].split(" ")))

    for word in spam[-1]:
        vocab.add(word)

vocab = list(vocab)
w2i = {word:i for word,i in enumerate(vocab)}

def to_indices(input, l=500):
    indices = list()

    for line in input:
        if len(line) < l:
            line = list(line) + ["<unk>"] * (l-len(line))

            idxs = list()

            for word in line:
                idxs.append(w2i[word])
            indices.append(idxs)
    return indices

spam_idx = to_indices(spam)
ham_idx = to_indices(ham)

train_spam_idx = spam_idx[0:-1000]
train_ham_idx = ham_idx[0:-1000]

test_spam_idx = spam_idx[-1000:]
test_ham_idx = ham_idx[-1000:]

train_data = list()
test_data = list()

train_target = list()
test_target = list()

for i in range(max(len(train_spam_idx), len(train_ham_idx))):

    train_data.append(train_spam_idx[i % len(train_spam_idx)])
    train_target.append([1])

    train_data.append(train_ham_idx[i % len(train_ham_idx)])
    train_target.append([0])

for i in range(max(len(test_spam_idx), len(test_ham_idx))):

    test_data.append(test_spam_idx[i % len(test_spam_idx)])
    test_target.append([1])

    test_data.append(test_ham_idx[i % len(test_ham_idx)])
    test_target.append([0])


def train(model, input_data)