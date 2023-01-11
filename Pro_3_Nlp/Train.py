from myModel import *
from DataPreProccisng import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import time
from chu_liu_edmonds import decode_mst

MLP_HIDDEN_DIM = 100
EPOCHS = 150
WORD_EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 25
HIDDEN_DIM = 125
LEARNING_RATE = 0.01
EARLY_STOPPING = 10  # num epochs with no validation acc improvement to stop training

ACCUMULATED_GRAD_STEPS = 50


def UAS_acc(true_tree, pred_tree):
    y_pred = pred_tree[0][1:]
    y_true = true_tree[1:].numpy()
    acc = (y_pred == y_true).sum() / len(y_true)
    return acc


def eval_model(score_mat):
    # score_mat, loss = model(sentence)
    predicted_tree = decode_mst(energy=score_mat.detach(), length=score_mat.shape[0], has_labels=False)
    return predicted_tree


def NLL_loss(tag_scores, true_tree_heads):
    loss_scores = tag_scores[:, 1:]
    clean_true_tree = torch.cat((torch.tensor([-1]), true_tree_heads[0][1:]))
    loss = loss_fun(loss_scores, clean_true_tree)
    return loss


def trainnn(model, train_data, test_data, epochs=10, batch_size=256, sequence_length=4):
    model.train()

    """
     :param epochs: number of epochs
     :param model_type: type of the model 'advanced' or 'base'
     :param test_epoch: test every test_epoch epochs
     :param save_model: save the model or not
     :param model_path: path to save the model
     :param save_plots: save the plot
     :param plot_dir: directory to save the plots in
     :param checkpoint_at_test: if True saves checkpoint of the model every test
     :param checkpoint_path: path to save the checkpoint to (appended the epoch number) if checkpoint_at_test==True
     :param time_run: if True rimes the run
     :return: the trained model
     """
    t = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    model.to(device)

    print("Training : ")
    best_accuracy = 0
    best_epoch = None
    # model.zero_grad()
    print(torch.autograd.set_detect_anomaly(True))
    for epoch in range(epochs):
        print(f"\n -- Epoch {epoch} --")
        train_accuracy, val_accuracy = train(model, device, optimizer, train_data, test_data)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
        if epoch - best_epoch == 3:
            break
    print(f"best accuracy: {best_accuracy:.6f} in epoch {best_epoch}")


def train(model, device, optimizer, train_dataset, val_dataset):
    accuracies = []
    acumulate_grad_steps = 128
    for phase in ["train", "validation"]:
        if phase == "train":
            model.train(True)
        else:
            model.train(False)  # or model.evel()
        correct = 0.0
        count = 0
        accuracy = None
        dataset = train_dataset if phase == "train" else val_dataset
        # t_bar = tqdm(dataset)
        for i, input_data in enumerate(dataset):
            words_idx_tensor, pos_idx_tensor, heads_tensor = input_data
            if phase == "train":
                words_idx_tensor, pos_idx_tensor, heads_tensor = input_data
                score_mat = model(input_data)

                pred_head = eval_model(score_mat)
                loss = NLL_loss(score_mat, heads_tensor)
                loss = loss / acumulate_grad_steps
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
            else:
                with torch.no_grad():
                    score_mat, _ = model(input_data)
                    pred_head = eval_model(score_mat)

            accuracy = UAS_acc(heads_tensor[0], pred_head)
            print(f"{phase} accuracy: {accuracy:.2f}")
        accuracies += [accuracy]
    return accuracies


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
    trainnn(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
