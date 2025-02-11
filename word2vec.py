import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA

import pdb

sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]
vocab = set()
for sentence in sentences:
    for word in sentence.split():
        vocab.add(word.lower())

vocab = list(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

class CustomDataset(Dataset):
    def __init__(self, sentences, window_size, word_to_ix, ix_to_word):
        self.sentences = sentences
        self.window_size = window_size
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word

        self.dataset = self.init_dataset()

    def init_dataset(self):
        dataset = []
        for sentence in self.sentences:
            sentence = sentence.lower().split()
            for i in range(len(sentence)-self.window_size + 1):
                input_data = sentence[i:i+self.window_size]
                dataset.append(input_data)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence = self.dataset[idx]
        sentence_ix = [self.word_to_ix[word] for word in sentence]

        # generate random number between 0 and window_size
        random_number = random.randint(0, window_size-1)
        input_data = sentence_ix[random_number]
        ran = random.randint(0, 1)
        if ran == 0:
            idx = random_number - 1
            if idx < 0:
                idx = random_number+1
            sentence_ix = sentence_ix[idx]
        else:
            idx = random_number + 1
            if idx == len(sentence_ix):
                idx = random_number-1
            sentence_ix = sentence_ix[idx]
        one_hot_input = F.one_hot(torch.tensor(input_data), num_classes=len(vocab))

        return  one_hot_input, torch.tensor(sentence_ix, dtype=torch.long)
    
window_size = 4
batch_size = 4
dataset = CustomDataset(sentences, window_size, word_to_ix, ix_to_word)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(vocab_size, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return x2

embedding_dim = 32
model = Net(len(vocab), embedding_dim).to("xpu")
loss_fn = nn.CrossEntropyLoss().to("xpu")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epoch = 1000

for i in range(epoch):
    total_loss = 0
    for one_hot_input, label in dataloader:
        optimizer.zero_grad()
        pred = model(one_hot_input.to(torch.float32).to("xpu"))
        loss = loss_fn(pred, label.to("xpu"))
        loss.backward()
        optimizer.step()
        # print(f"Loss: {loss.item()}")
        total_loss += loss.item()
    print(f"Total loss: {total_loss}")

embeddings = model.fc1.weight.data.cpu().numpy().T
if embeddings.shape[0] > 2:
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
else:
    embeddings_2d = embeddings

# Plotting the 2D embeddings with labels.
plt.figure(figsize=(10,10))
for i, word in enumerate(vocab):
    x, y = embeddings_2d[i, 0], embeddings_2d[i, 1]
    plt.scatter(x, y)
    plt.annotate(word, (x, y), textcoords="offset points", xytext=(5,5), ha='center')
plt.title("2D Visualization of Word Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
