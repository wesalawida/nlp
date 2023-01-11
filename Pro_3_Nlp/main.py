from collections import defaultdict
# from torchtext.vocab import Vocab
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from myModel import  *
from DataPreProccisng import *
from chu_liu_edmonds import *
from os import path
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

import time
from chu_liu_edmonds import decode_mst
import matplotlib.pyplot as plt
# taken from the paper
MLP_HIDDEN_DIM = 100
EPOCHS = 150
WORD_EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 25
HIDDEN_DIM = 125
LEARNING_RATE = 0.01
EARLY_STOPPING = 10  # num epochs with no validation acc improvement to stop training

ACCUMULATED_GRAD_STEPS = 50  # This is the actual batch_size, while we officially use batch_size=1
PATH = "./basic_model_best_params"
use_validation = False


cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# def NLLL_function(scores, true_tree):
#     """
#     Parameters
#     ----------
#     scores - a matrix of size (sentence_length x sentence length)
#     true_tree - ground truth dependency tree
#
#     Returns the loss
#     -------
#     """
#     clean_scores = scores[:, 1:]                # ROOT cant be modifier
#     clean_true_tree = true_tree[1:]
#     sentence_length = clean_scores.shape[1]     # without root
#     loss = 0
#     for mod in range(sentence_length):
#         loss += cross_entropy_loss(clean_scores[:, mod].unsqueeze(dim=0), clean_true_tree[mod:mod+1])
#     return (1.0/sentence_length) * loss
#
#
#
# def accuracy(ground_truth, energy_table):
#     predicted_mst, _ = decode_mst(energy=energy_table.detach(), length=energy_table.shape[0], has_labels=False)
#     # first one is the HEAD of root so we avoid taking it into account
#     y_pred = torch.from_numpy(predicted_mst[1:])
#     y_true = ground_truth[1:]
#     acc = (y_pred == y_true).sum()/float(y_true.shape[0])
#     return acc.item()
#
#
# def evaluate(model, data_loader):
#     val_acc = 0
#     val_size = 0
#     for batch_idx, input_data in enumerate(data_loader):
#         val_size += 1
#         with torch.no_grad():
#             words_idx_tensor, pos_idx_tensor, heads_tensor = input_data
#             tag_scores = model(input_data)
#             #tag_scores = model(words_idx_tensor, pos_idx_tensor)
#             val_acc += (accuracy(heads_tensor[0].cpu(), tag_scores.cpu()))
#     return val_acc / val_size
def loss_function(scores, real):
    nll_loss = nn.NLLLoss(ignore_index=-1)
    #log_soft_max = nn.LogSoftmax(dim=1)
    cur_loss = nll_loss(scores, real)
    return cur_loss

def evaluate(model, data_loader):
    with torch.no_grad():
        accuracy = 0
        words_count = 0
        accumulated_loss = 0
        for i, sentence in enumerate(data_loader):
            words_idx_tensor, pos_idx_tensor, _, true_tree_heads = sentence

            score_matrix = model((words_idx_tensor, pos_idx_tensor, true_tree_heads))
            accumulated_loss += loss_function(score_matrix, true_tree_heads.to(device)).item()

            score_matrix_numpy = score_matrix.squeeze(0).cpu().detach().numpy()
            predicted_tree, _ = decode_mst(score_matrix_numpy, len(score_matrix_numpy), has_labels=False)
            true_tree_heads = true_tree_heads.squeeze(0).cpu().detach().numpy()
            accuracy += np.sum(true_tree_heads[1:] == predicted_tree[1:]) # compare without first word
            words_count += len(predicted_tree)

        accumulated_loss = accumulated_loss / len(data_loader)
        accuracy = accuracy / words_count

    return accuracy, accumulated_loss
def main():
    # sanity check
    train_path = "data/train.labeled"
    test_path = "data/test.labeled"
    PATH_TO_SAVE_MODEL = "data/basic/basic_model.pkl"
    x = data_Preprocessing(train_path, test_path)
    train_dataloader = DataLoader(x.train_pro_dataset, shuffle=True)
    test_dataloader = DataLoader(x.test_pro_dataset, shuffle=False)

    word_vocab_size = len(x.word_to_index)
    print(word_vocab_size)
    tag_vocab_size = len(x.pos_counter)
    print(tag_vocab_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = DependencyParser(word_vocab_size, word_embedding_dim, pos_vocab_size, pos_embedding_dim, hidden_LSTM,
                             hidden_MLP).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training start
    print("Training Started")
    model.train()
    train_accuracy_list = []
    test_accuracy_list = []

    train_loss_list = []
    test_loss_list = []

    loss_list = []
    epochs = EPOCHS

    for epoch in range(epochs):
        epoch_start_time = time.time()
        acc_list = []  # to keep track of accuracy
        printable_loss = 0  # To keep track of the loss value
        i = 0
        for batch_idx, input_data in enumerate(train_dataloader):
            i += 1
            words_idx_tensor, pos_idx_tensor,  true_tree_heads = input_data

            score_matrix = model(input_data)

            loss = loss_function(score_matrix, true_tree_heads.to(device))
            printable_loss += loss.item()
            loss.backward()

            if i % ACCUMULATED_GRAD_STEPS == 0 and i > 0:
                optimizer.step()
                model.zero_grad()
        printable_loss = printable_loss / len(x.train_pro_dataset)
        loss_list.append(float(printable_loss))
        print("evaluating epoch:")
        test_accuracy, test_loss = evaluate(model, test_dataloader)
        train_accuracy, train_loss = evaluate(model, train_dataloader)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)

        e_interval = i

        time_of_epoch = time.time() - epoch_start_time
        epoch_print = "---\nEpoch {} Completed,\tTrain Loss {}\tTrain Accuracy: {}\t Test Loss {}\tTest Accuracy: {}\t \
        time: {}".format(epoch + 1, train_loss, train_accuracy, test_loss, test_accuracy, time_of_epoch)
        print(epoch_print)

    # save model and vocabulary
    print(
        "saving model: \nvocabulary path: data/basic/word_vocabulary.pkl, data/basic/pos_vocabulary.pkl \nmodel file: {}".format(
            PATH_TO_SAVE_MODEL))
    torch.save(model, PATH_TO_SAVE_MODEL)
    with open('data/basic/word_vocabulary.pkl', 'wb+') as output:
        pickle.dump(x.word_counter, output, pickle.HIGHEST_PROTOCOL)

    with open('data/basic/pos_vocabulary.pkl', 'wb+') as output:
        pickle.dump(x.pos_counter, output, pickle.HIGHEST_PROTOCOL)

    # show graphs:

    # train:
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle('Train results - Basic model', fontsize=16)
    axs[0].plot(np.arange(len(train_accuracy_list)), train_accuracy_list, c="red", label="Accuracy")
    axs[0].set_title('Train accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Value')

    axs[1].plot(np.arange(len(train_loss_list)), train_loss_list, c="blue", label="Loss")
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Value')
    axs[1].set_title('Train loss')

    plt.show()

    # test:
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle('Test results - Basic model', fontsize=16)
    axs[0].plot(np.arange(len(test_accuracy_list)), test_accuracy_list, c="red", label="Accuracy")
    axs[0].set_title('Test accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Value')

    axs[1].plot(np.arange(len(test_loss_list)), test_loss_list, c="blue", label="Loss")
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Value')
    axs[1].set_title('Test loss')

    plt.show()






















    # # Define the loss function as the Negative Log Likelihood loss (NLLLoss)
    # loss_function = nn.NLLLoss()
    #
    # # We will be using a simple SGD optimizer to minimize the loss function
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # acumulate_grad_steps = 128  # This is the actual batch_size, while we officially use batch_size=1
    #
    # # Training start
    # print("Training Started")
    # epoch_loss_list = []
    # epoch_train_acc_list = []
    # epoch_test_acc_list = []
    # best_val_acc = 0
    # num_epochs_wo_improvement = 0
    # for epoch in range(EPOCHS):
    #     val_acc = evaluate(model, test_dataloader)
    #     print("EPOCH = ", epoch)
    #     print("EPOCH test acc = ", val_acc)
    #     if val_acc < best_val_acc:     # no improvement
    #         num_epochs_wo_improvement += 1
    #         if num_epochs_wo_improvement >= EARLY_STOPPING:
    #             print("STOPPED TRAINING DUE TO EARLY STOPPING")
    #             return
    #     else:                                   # improvement
    #         print("saving model since it improved on test :)")
    #         torch.save(model.state_dict(), PATH)
    #         num_epochs_wo_improvement = 0
    #         best_val_acc = val_acc
    #         # fig = plt.figure()
    #         # plt.subplot(3, 1, 1)
    #         # plt.plot(epoch_loss_list)
    #         # plt.title("loss")
    #         # plt.subplot(3, 1, 2)
    #         # plt.plot(epoch_train_acc_list)
    #         # plt.title("train UAS")
    #         # plt.subplot(3, 1, 3)
    #         # plt.plot(epoch_test_acc_list)
    #         # plt.title("test UAS")
    #         # print(epoch_train_acc_list)
    #         # plt.savefig('./basic_model_graphs.png')
    #
    #     # train
    #     acc = 0  # to keep track of accuracy
    #     printable_loss = 0  # To keep track of the loss value
    #     i = 0
    #     batch_loss = 0
    #     batch_acc = 0
    #     epoch_loss = 0
    #
    #     for batch_idx, input_data in enumerate(train_dataloader):
    #         i += 1
    #         words_idx_tensor, pos_idx_tensor, heads_tensor = input_data
    #
    #         tag_scores = model(input_data)
    #         loss = NLLL_function(tag_scores, heads_tensor[0].to(device))
    #         # epoch statistics
    #         epoch_loss += loss
    #         #
    #         loss = loss / acumulate_grad_steps
    #         loss.backward()
    #         batch_loss += loss
    #         acc = (accuracy(heads_tensor[0].cpu(), tag_scores.cpu())) / acumulate_grad_steps
    #         batch_acc += acc
    #         if i % acumulate_grad_steps == 0:
    #             optimizer.step()
    #             model.zero_grad()
    #             print("batch_loss = ", batch_loss.item())
    #             print("batch_acc = ", batch_acc)
    #             batch_loss = 0
    #             batch_acc = 0
    #     # end of epoch - get statistics
    #     epoch_loss_list.append(epoch_loss / i)
    #     epoch_train_acc_list.append(evaluate(model, train_dataloader))
    #     epoch_test_acc_list.append(evaluate(model, test_dataloader))
    # print("batch done ! ")
    # # end of train - plot the two graphs
    # # fig = plt.figure()
    # # plt.subplot(3, 1, 1)
    # # #plt.plot(epoch_loss_list.detach().numpy())
    # # plt.plot(epoch_loss_list)
    # # plt.title("loss")
    # # plt.subplot(3, 1, 2)
    # # plt.plot(epoch_train_acc_list)
    # # plt.title("train UAS")
    # # plt.subplot(3, 1, 3)
    # # plt.plot(epoch_test_acc_list)
    # # plt.title("test UAS")
    # # plt.show()
    # # plt.savefig('basic_model_graphs.png')


if __name__ == "__main__" :
        main()