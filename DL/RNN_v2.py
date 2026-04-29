import re
from nltk.tokenize import word_tokenize
from MLFrameWork import *

np.random.seed(1)

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

tokens = [clean_text_nltk(review) for review in raw]

new_tokens = list()
for line in tokens:
    new_tokens.append(['-'] * (6 - len(line)) + line)
tokens = new_tokens

vocab = set()

for sent in tokens:
    for word in sent:
        vocab.add(word)

vocab = list(vocab)

word2index = {word: i for i, word in enumerate(vocab)}

indices = list()
for line in tokens:
    idx = list()
    for w in line:
        idx.append(word2index[w])
    indices.append(idx)

data = np.array(indices)

embed = Embedding(len(vocab), 16)

model = RNNCell(16,16,len(vocab))

criterion = CrossEntropyLoss()

optim = SGD(model.get_tensors() + embed.get_tensors(), 0.05)

for iter in range(1000):
    batch_size = 100
    total_loss = 0

    hidden = model.init_hidden(batch_size)

    for t in range(5):
        input = Tensor(data[0:batch_size,t], autograd=True)

        rnn_input = embed.forward(input)

        output, hidden = model.forward(rnn_input, hidden)

    target = Tensor(data[0:batch_size, t+1], autograd=True)

    loss = criterion.forward(output, target)
    loss.backward()

    optim.step()

    total_loss += loss.data

    if(iter % 200 == 0):
        p_correct = (target.data == np.argmax(output.data,axis=1)).mean()
        print_loss = total_loss / (len(data)/batch_size)
        print("Loss: ", print_loss, "% Correct: ", p_correct)

# Тест
batch_size = 1
hidden = model.init_hidden(batch_size)

for t in range(5):
    input = Tensor(data[0:batch_size,t], autograd=True)
    rnn_input = embed.forward(input)
    output, hidden = model.forward(rnn_input, hidden)

target = Tensor(data[0:batch_size, t+1], autograd=True)
loss = criterion.forward(output, target)
ctx = ""

for i in data[0:batch_size][0][0:-1]:
    ctx += vocab[i] + " "

print("Context:",ctx)
print("Pred:", vocab[output.data.argmax()])