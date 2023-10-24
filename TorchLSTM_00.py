import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


'''
based on
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
'''


torch.manual_seed(1)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    # Tags are: DET - determiner; NN - noun; V - verb
    # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
# For each words-list (sentence) and tags-list in each tuple of training_data
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index


EMBEDDING_DIM = 6
HIDDEN_DIM = 6
VOCAB_SIZE = len(word_to_ix)
OUTPUT_SIZE = len(tag_to_ix)


class CustomLSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=128,
            num_layers=2
        )
        self.output = nn.Linear(128, OUTPUT_SIZE)

    def forward(self, x):
        z = self.embedding(x)
        lstm_out, _ = self.lstm(z.view(len(x), 1, -1))
        out = self.output(lstm_out.view(len(x), -1))
        out = F.log_softmax(out, dim=1)
        return out


lstm = CustomLSTM()
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(lstm.parameters(), lr=1/10)


with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    scores = lstm(inputs)
    print(scores)



























































































































































































































































































































































































