import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

context_size = 2
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

data = []
for i in range(context_size, len(raw_text) - context_size):
    context = [
        raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]
    ]
    target = raw_text[i]
    data.append((context, target))


class CBOW(nn.Module):
    def __init__(self, n_word, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(n_word, embedding_dim)
        self.project = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, n_word)

    def forward(self, x):
        x = self.embedding(x)
        x = self.project(x)
        # notice
        x = torch.sum(x, 0, keepdim=True)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        x = F.log_softmax(x)
        return x


model = CBOW(len(word_to_idx), 100, context_size)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(100):
    print('epoch {}'.format(epoch))
    print('*' * 10)
    running_loss = 0
    for (word, target) in data:
        word = torch.tensor([word_to_idx[i] for i in word], dtype=torch.long)
        word = word.to(device)

        target = torch.tensor([word_to_idx[target]], dtype=torch.long)
        target = target.to(device)

        # forward
        out = model(word)
        loss = criterion(out, target)
        running_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('loss: {:.6f}'.format(running_loss / len(data)))

word, target = data[1]
word = torch.tensor([word_to_idx[i] for i in word], dtype=torch.long)
word = word.to(device)
out = model(word)
_, pred = torch.max(out, 1)
pred_word = idx_to_word[pred.item()]
print("Real are about {} study the , predict  are about {} study the".format(target, pred_word))
