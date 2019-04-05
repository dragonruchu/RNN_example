import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

context_size = 2
embedding_dim = 10

# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigram = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2])
           for i in range(len(test_sentence) - 2)]


vocb = set(test_sentence)
word_to_idx = {word: i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

res = []


class NgramModel(nn.Module):
    def __init__(self, vocb_size, context_size, embedding_dim):
        super(NgramModel, self).__init__()
        self.n_word = vocb_size
        self.embedding = nn.Embedding(self.n_word, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        # print(emb)
        res.append(emb)
        emb = emb.view(1, -1)
        out = F.relu(self.linear1(emb))
        out = self.linear2(out)
        log_prob = F.log_softmax(out)
        return log_prob


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NgramModel(len(word_to_idx), context_size, 100)
model = model.to(device)
# print(model)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)


for epoch in range(10):
    print('epoch: {}'.format(epoch + 1))
    print("*" * 10)

    running_loss = 0
    for (word, target) in trigram:
        num_context = [word_to_idx[i] for i in word]
        word = torch.tensor(num_context, dtype=torch.long)
        word = word.to(device)

        num_target = [word_to_idx[target]]
        target = torch.tensor(num_target, dtype=torch.long)
        target = target.to(device)

        # forward
        out = model(word)
        loss = criterion(out, target)
        running_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Loss : {:.6f}".format(running_loss / len(word_to_idx)))

# word, target = trigram[3]
# word = torch.tensor([word_to_idx[i] for i in word], dtype=torch.long)
# word = word.to(device)
# out = model(word)
# _, pred = torch.max(out, 1)
# pred_word = idx_to_word[pred.item()]
# print("Real word is {}, predict word is {}".format(target, pred_word))
# print(res)
# print(len(res), len(res[0]))
# print(len(trigram))
# for i in range(0, 1130, 113):
#     print(res[i])
