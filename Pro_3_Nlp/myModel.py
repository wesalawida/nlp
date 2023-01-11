import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')
loss_fun = nn.NLLLoss(ignore_index=-1, reduction='mean')


# def NLLL_function(scores, true_tree):
#
#     clean_scores = scores[:, 1:]
#     clean_true_tree = torch.cat((torch.tensor([-1]), true_tree[1:]))
#     loss = loss_fun(clean_scores, clean_true_tree)
#
#     return loss




class DependencyParser(nn.Module):
    def __init__(self, word_vocab_size, word_embedding_dim, pos_vocab_size, pos_embedding_dim, hidden_LSTM, hidden_MLP):
        super(DependencyParser, self).__init__()
        self.word_embedding = nn.Embedding(word_vocab_size,
                                           word_embedding_dim)  # Implement embedding layer for words (can be new or pretrained - word2vec/glove)
        self.pos_embedding = nn.Embedding(pos_vocab_size,
                                          pos_embedding_dim)  # Implement embedding layer for words (can be new or pretrained - word2vec/glove)
        self.hidden_dim = self.word_embedding.embedding_dim
        self.together = word_embedding_dim + pos_embedding_dim
        self.lstm = nn.LSTM(input_size=self.together, hidden_size=hidden_LSTM, num_layers=2, bidirectional=True,
                            batch_first=False)
        self.layer1 = nn.Linear(2*hidden_LSTM, hidden_MLP)  # to know what the in put of this
        self.layer2 = nn.Linear(2*hidden_LSTM, hidden_MLP)
        self.layer3 = nn.Linear(hidden_MLP, 1)

        self.tanh = nn.Tanh()

        # self.encoder =  # Implement BiLSTM module which is fed with word embeddings and outputs hidden representations
        # self.edge_scorer =  # Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph
        self.loss_function = nn.CrossEntropyLoss(reduction='mean')
        self.nll_loss = nn.NLLLoss(ignore_index=-1, reduction='mean')

    def forward(self, sentence):
        word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence

        # Pass word_idx through their embedding layer
        word_idx = self.word_embedding(word_idx_tensor)
        pos_idx = self.pos_embedding(pos_idx_tensor)
        embeds = torch.cat((word_idx, pos_idx), 2)
        # Get Bi-LSTM hidden representation for each word in sentence
        # out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))
        # out will have shape (sequence_length, batch_size, hidden_size)
        lstm_out = out.view(out.shape[0], -1)
        sentence_length = lstm_out.shape[0]
        heads_hidden = self.layer1(lstm_out)
        mods_hidden = self.layer2(lstm_out)

        # Get score for each possible edge in the parsing graph, construct score matrix
        tag_scores = torch.zeros(size=(sentence_length, sentence_length))
        # tag_scores = torch.zeros((sentence_length, sentence_length), dtype=torch.float32)
        #
        # # set the diagonal values of the score matrix to negative infinity
        # #tag_scores = tag_scores.masked_fill(torch.eye(sentence_length, dtype=torch.uint8), -float('inf'))
        for mod in range(sentence_length):
            mod_hidden = mods_hidden[mod]
            summed_values = mod_hidden + heads_hidden  # a single mod with all heads possibilities
            x =  self.tanh(summed_values)
            tag_scores[:, mod] = torch.flatten(self.layer3(x))
            tag_scores[mod, mod] = -np.inf  # a word cant be its head


        # for idx in range(sentence_length):
        #     mod_hidden = mods_hidden[idx]  # shape: (batch_size, hidden_size)
        #     summed_values = mod_hidden + heads_hidden  # shape: (batch_size, sentence_length, hidden_size)
        #     x = torch.tanh(summed_values)  # shape: (batch_size, sentence_length, hidden_size)
        #     x = self.layer3(x)  # shape: (batch_size, sentence_length, output_size)
        #     tag_scores[idx, :] = torch.flatten(x)



        #todo : check the loss softmax
        #tag_scores = F.log_softmax(tag_scores, dim=1)  # [seq_length, tag_dim]
        # loss_scores = tag_scores[:, 1:]
        # clean_true_tree = torch.cat((torch.tensor([-1]), true_tree_heads[0][1:]))
        # loss = loss_fun(loss_scores, clean_true_tree)
        return tag_scores


word_vocab_size = 15948
word_embedding_dim = 100
pos_vocab_size = 47
pos_embedding_dim = 25
hidden_LSTM = 125
hidden_MLP = 100
model = DependencyParser(word_vocab_size, word_embedding_dim, pos_vocab_size, pos_embedding_dim, hidden_LSTM,
                         hidden_MLP)

print(model)
