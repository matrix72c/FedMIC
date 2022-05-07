import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset
import os


class Config:
    """
    General configuration
    """
    train_data_path = "data/ml-100k.train.rating"
    test_data_path = "data/ml-100k.test.negative"
    factor_num = 32
    num_layers = 3
    dropout = 0
    learning_rate = 0.001
    batch_size = 64
    epochs = 10
    dropout = 0
    device = None
    rounds = 40000
    top_k = 10
    sample_size = 10
    model = "NeuMF"
    neg_pos_ratio = 4
    eval_every = 50
    distill_batch_size = 256
    distill_learning_rate = 0.001
    distill_epochs = 5
    distill_loss_threshold = 1e-5
    distill_lr_step = 90
    distill_lr_decay = 0.95
    distill_pos_ratio = 0.5
    distill_T = 3
    lr_step = epochs
    lr_decay = 0.95
    mu = 0.2
    result_path = "./result/distill_"


def get_ncf_data():
    """
    Load data from files, and convert them into [[user_id, item_id], 0/1] format.
    """
    train_data = pd.read_csv(Config.train_data_path, sep='\t', header=None, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1
    pos_num = len(train_data)
    train_data = train_data.values.tolist()
    train_label = [1 for _ in range(pos_num)]

    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    neg_ui = []
    for user, item in train_data:
        train_mat[user, item] = 1.0
        for t in range(Config.neg_pos_ratio):
            j = np.random.randint(item_num)
            while (user, j) in train_mat:
                j = np.random.randint(item_num)
            neg_ui.append([user, j])
    train_data += neg_ui
    train_label += [0 for _ in range(pos_num * Config.neg_pos_ratio)]

    with open(Config.test_data_path, "r") as f:
        lines = f.readlines()
    test_list = [[] for _ in range(len(lines))]
    ret_test = []
    for line in lines:
        tmp_str_list = line.split("\t")
        user_str, item_str = tmp_str_list[0][1:-1].split(",")
        item_list = [int(i) for i in tmp_str_list[1:]]
        user = int(user_str)
        gt_item = int(item_str)
        item_list.append(gt_item)
        test_list[user] = [item_list, gt_item]
    for i in range(len(test_list)):
        tmp = []
        for j in range(len(test_list[i][0])):
            tmp.append([i, test_list[i][0][j]])
        ret_test.append([tmp, test_list[i][1]])
    return user_num, item_num, train_data, train_label, ret_test


def distribute_data(train_data, train_label, user_num):
    """
    Distribute the data into each user client(each user has an independent client).
    """
    clients_train_data, clients_train_label = [[] for _ in range(user_num)], [[] for _ in range(user_num)]
    for i in range(len(train_data)):
        user, item = train_data[i][0], train_data[i][1]
        clients_train_data[user].append([user, item])
        clients_train_label[user].append(train_label[i])
    return clients_train_data, clients_train_label


def hit(gt_item, pred_items):
    """
    Hit rate
    """
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    """
    Normalized discounted cumulative gain
    """
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def evaluate(model, test_datas):
    """
    Evaluate the NCFModel
    """
    model = model.to(Config.device)
    model.eval()
    hits = []
    ndcgs = []
    for test_data in test_datas:
        x = torch.tensor(test_data[0]).to(torch.long).to(Config.device)
        gt_item = test_data[1]
        prediction = model(x)
        _, indices = torch.topk(prediction, Config.top_k)
        recommends = torch.take(x[:, 1], indices).cpu().numpy().tolist()
        hits.append(hit(gt_item, recommends))
        ndcgs.append(ndcg(gt_item, recommends))
    return np.mean(hits), np.mean(ndcgs)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def torch_delete(tensor, indices):
    mask = torch.ones(len(tensor), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


class NCFDataset(Dataset):
    """
    NCF dataset for PyTorch DataLoader
    """

    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)


class Logger():
    """
    Logger for train and test process
    """

    def __init__(self):
        self.client_df = pd.DataFrame(columns=['client_id', 'local_epoch', 'loss'])
        self.server_df = pd.DataFrame(columns=['round', 'distill_loss', 'hr@10', 'ndcg@10'])
        self.client_df.to_csv(Config.result_path + "client.csv", index=False)
        self.server_df.to_csv(Config.result_path + "server.csv", index=False)

    def log_client_loss(self, client_id, local_epoch, loss):
        self.client_df.loc[len(self.client_df)] = [client_id, local_epoch, loss]
        if self.client_df.shape[0] % 10000 == 0:
            self.client_df.to_csv(Config.result_path+"client.csv", index=False, header=False, mode='a')
            self.client_df.drop(self.client_df.index, inplace=True)

    def log_distill_result(self, rnd, distill_loss, hr, hdcg):
        self.server_df.loc[len(self.server_df)] = [rnd, distill_loss, hr, hdcg]
        if self.server_df.shape[0] % 100 == 0:
            self.server_df.to_csv(Config.result_path+"server.csv", index=False, header=False, mode='a')
            self.server_df.drop(self.server_df.index, inplace=True)