import sys
from MLFrameWork import *

np.random.seed(0)

f = open('../data/shakespear.txt','r')
raw = f.read()
f.close()

vocab = list(set(raw))

word2index = {word: i for i, word in enumerate(vocab)}

indices = np.array(list(map(lambda x: word2index[x], raw)))

embed = Embedding(vocab_size=len(vocab), dim=512)
model = LSTMCell(n_in=512, n_hide=512, n_out=len(vocab))
model.w_ho.weight.data *= 0

criterion = CrossEntropyLoss()

optim = SGD(tensors=model.get_tensors() + embed.get_tensors(), alpha=0.05)

def generate_sample(n=30, init_char=' '):
    s = ""
    hidden = model.init_hidden(batch_size=1)
    input = Tensor(np.array([word2index[init_char]]))
    for i in range(n):
        rnn_input = embed.forward(input)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)
        m = output.data.argmax()
        c = vocab[m]
        input = Tensor(np.array([m]))
        s += c
    return s

batch_size = 16
bptt = 25
n_batches = int((indices.shape[0] / (batch_size)))

trimmed_indices = indices[:n_batches*batch_size]
batched_indices = trimmed_indices.reshape(batch_size, n_batches).transpose()

input_batched_indices = batched_indices[0:-1]
target_batched_indices = batched_indices[1:]

n_bptt = int(((n_batches-1) / bptt))

input_batches = input_batched_indices[:n_bptt*bptt].reshape(n_bptt, bptt, batch_size)
target_batches = target_batched_indices[:n_bptt*bptt].reshape(n_bptt, bptt, batch_size)

def train(iterations=100):
    for iter in range(iterations):
        total_loss, n_loss = 0, 0

        hidden = model.init_hidden(batch_size)

        for batch_i in range(len(input_batches)):

            losses = list()

            hidden = (Tensor(hidden[0].data, autograd=True), Tensor(hidden[1].data, autograd=True))

            for t in range(bptt):
                input = Tensor(input_batches[batch_i][t], autograd=True)

                rnn_input = embed.forward(input)

                output, hidden = model.forward(rnn_input, hidden)

                target = Tensor(target_batches[batch_i][t], autograd=True)

                batch_loss = criterion.forward(output, target)

                if t == 0:
                    losses.append(batch_loss)
                else:
                    losses.append(batch_loss + losses[-1])

            losses[-1].backward()
            optim.step()

            total_loss += losses[-1].data / bptt
            epoch_loss = np.exp(total_loss / (batch_i + 1))

            if batch_i % 5 == 0:
                log = "\r Iter:" + str(iter)
                log += " - Alpha:" + str(optim.alpha)[0:5]
                log += " - Batch "+str(batch_i)+"/"+str(len(input_batches))
                log += " - Loss:" + str(epoch_loss)
                log += " - " + generate_sample(n=30, init_char='\n').replace("\n"," ")
                sys.stdout.write(log)

        optim.alpha *= 0.99
        print()

train(10)