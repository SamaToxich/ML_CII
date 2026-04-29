import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from torch.amp import GradScaler, autocast

# Включите cuDNN автоподбор алгоритмов
torch.backends.cudnn.benchmark = True
# Оптимизация для LSTM
torch.backends.cudnn.enabled = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Функция генерации текста
def generate_sample(model, n=30, init_char=' ', temperature=1.0):
    model.eval()
    with torch.no_grad():
        s = ""
        hidden = model.init_hidden(1)
        input_idx = torch.tensor([[w2i[init_char]]]).to(device)

        for i in range(n):
            output, hidden = model(input_idx, hidden)
            output = output[0, -1, :] / temperature  # Последний символ в последовательности
            probs = torch.softmax(output, dim=0)
            m = torch.multinomial(probs, 1).item()
            c = i2w[m]
            input_idx = torch.tensor([[m]]).to(device)
            s += c
    model.train()
    return s

# Загрузка данных
with open('../data/shakespear.txt', 'r') as f:
    raw = f.read()

vocab = list(set(raw))
vocab_size = len(vocab)

w2i = {word: i for i, word in enumerate(vocab)}
i2w = {i: word for word, i in w2i.items()}

indices = np.array([w2i[ch] for ch in raw])


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024, num_layers=1):
        super(CharLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size,embed_size)

        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

        self.fc.weight.data.fill_(0)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x) # (batch_size, seq_length, embedding_dim)

        if hidden is None:
            output, hidden = self.lstm(x)
        else:
            output, hidden = self.lstm(x, hidden)

        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, 1024).to(device),
                torch.zeros(1, batch_size, 1024).to(device))


model = CharLSTM(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)
scaler = GradScaler('cuda')

# Подготовка данных
batch_size = 16
bptt = 25

indices_tensor = torch.tensor(indices, dtype=torch.long)

n_batches = indices_tensor.size(0) // batch_size
trimmed_indices = indices_tensor[:n_batches * batch_size]
batched_indices = trimmed_indices.view(batch_size, -1).t().contiguous() # (n_batches, batch_size)

input_batched_indices = batched_indices[:-1]
target_batched_indices = batched_indices[1:]

n_bptt = (n_batches - 1) // bptt

input_batches = input_batched_indices[:n_bptt * bptt].view(n_bptt, bptt, batch_size)
target_batches = target_batched_indices[:n_bptt * bptt].view(n_bptt, bptt, batch_size)

# Перенос данных на GPU
input_batches = input_batches.to(device)
target_batches = target_batches.to(device)

def train(iterations=500):

    for epoch in range(iterations):
        total_loss = 0

        hidden = model.init_hidden(batch_size)

        for batch_i in range(len(input_batches)):
            if hidden[0].requires_grad:
                hidden = (hidden[0].detach(), hidden[1].detach())

            optimizer.zero_grad()

            input_seq = input_batches[batch_i].t().contiguous() # (batch_size, bptt)
            target_seq = target_batches[batch_i].t().contiguous()  # (batch_size, bptt)

            # Использование mixed precision
            with autocast('cuda'):
                output, hidden = model(input_seq, hidden)
                output = output.view(-1, vocab_size)
                target_seq = target_seq.view(-1)
                loss = criterion(output, target_seq)

            # Scale loss and backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            avg_loss = np.exp(total_loss / (batch_i + 1))
        # Уменьшаем learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.99

        if epoch % 10 == 0:
            log = f"\r Epoch: {epoch}"
            log += f" - LR: {optimizer.param_groups[0]['lr']:.5f}"
            log += f" - Loss: {avg_loss:.4f}"

            # Генерация сэмпла
            sample = generate_sample(model, n=30, init_char='\n')
            sample = sample.replace("\n", " ")
            log += f" - {sample}"

            sys.stdout.write(log[:200])
            sys.stdout.flush()


train()
print(generate_sample(model, n=1000, init_char='\n'))