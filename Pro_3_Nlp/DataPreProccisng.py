import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader


UNKNOWN_TOKEN = "<unk>"
ROOT_TOKEN = "<root>"


class data_Preprocessing:

    def __init__(self, train_path, test_path):
        # here data from all the train and test data
        self.train_path = train_path
        self.test_path = test_path
        self.word_counter = {}
        self.pos_counter = {}
        self.word_to_index = {}
        self.word_count = 2
        self.pos_count = 2
        self.word_emb(self.train_path)
        self.word_emb(self.test_path)
        self.len_word2ind = len(self.word_to_index)

        # start with every data alone
        self.train_sentences = []
        self.data2tuple(train_path, 'train')
        self.train_pro_dataset = self.Data_pros(self.train_sentences)
        self.train_dataloader = DataLoader(self.train_pro_dataset, shuffle=True)

        # start with every data alone
        self.test_sentences = []
        self.data2tuple(test_path, 'test')
        self.test_pro_dataset = self.Data_pros(self.test_sentences)
        self.test_dataloader = DataLoader(self.test_pro_dataset, shuffle=False)
        print("done")

    def word_emb(self, path):
        # Create an empty dictionary
        self.word_to_index[ROOT_TOKEN] = 0
        self.word_to_index[UNKNOWN_TOKEN] = 1
        self.pos_counter[ROOT_TOKEN] = 0
        self.pos_counter[UNKNOWN_TOKEN] = 1

        with open(path) as f:
            for line in f:
                s_list = self.split_string(line)[0]
                if (len(s_list) != 10):
                    continue
                word, pos = s_list[1], s_list[3]
                if word not in self.word_to_index.keys():
                    self.word_to_index[word] = self.word_count
                    self.word_count += 1
                if word in self.word_counter.keys():
                    self.word_counter[word] += 1
                else:
                    self.word_counter[word] = 1
                if pos not in self.pos_counter.keys():
                    self.pos_counter[pos] = self.pos_count
                    self.pos_count += 1

    def split_string(self, string):
        # Split the string on the newline character
        lines = string.split('\n')

        # Split each line on the tab character
        lines = [line.split('\t') for line in lines]
        return lines

    def data2tuple(self, path, datt_type):
        """main reader function which also populates the class data structures"""
        with open(path) as f:
            curr_s = []
            for line in f:
                s_list = self.split_string(line)[0]
                # splited_words is of length 10 if we are still in the same sentence
                if len(s_list) == 10:
                    curr_s.append((s_list[1], s_list[3], s_list[6]))
                else:
                    if datt_type == 'train':
                        self.train_sentences.append(curr_s)
                    else:
                        self.test_sentences.append(curr_s)
                    curr_s = []

    def Data_pros(self, sant_lists):
        all_sentences = {}
        x = self.pos_counter['<root>']
        for i, sen in enumerate(sant_lists):
            word_list = []
            pos_list = []
            head_list = []
            word_list.append(0)
            pos_list.append(x)
            head_list.append(-1)
            for word, pos, head in sen:
                prob = 0.25 / (self.word_counter[word] + 0.25)
                word_list.append(
                    1 if np.random.random() < prob or word not in self.word_to_index else self.word_to_index[word])
                pos_list.append(self.pos_counter.get(pos))
                head_list.append(-1 if head == '_' else int(head))
            all_sentences[i] = (
                torch.tensor(word_list, dtype=torch.long, requires_grad=False),
                torch.tensor(pos_list, dtype=torch.long, requires_grad=False),
                torch.tensor(head_list, dtype=torch.long, requires_grad=False)
            )
        return all_sentences

    def dataloder(self):
        train_dataloader = DataLoader(self.train_pro_dataset, shuffle=True)
        test_dataloader = DataLoader(self.test_pro_dataset, shuffle=False)
        return  train_dataloader,test_dataloader

