import math

import numpy as np
import torch
import torch.nn.functional as F
from model import NCFModel
from config import Config


class Client:
    def __init__(self, train_data, test_data, user_num, item_num):
        self.train_data = train_data
        self.test_data = test_data
        self.user_num = user_num
        self.item_num = item_num
        self.model = NCFModel(user_num, item_num)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.learning_rate)

    def train_batch(self, x, y):
        y_ = self.model(x)
        mask = (y > 0).float()
        loss = torch.nn.functional.mse_loss(y_ * mask, y)  # only calculate loss for positive samples
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), y_.detach()

    def train(self, rnd=0):
        self.model.join_output_weights()  # TODO: why?
        epochs = Config.epochs
        batch_size = Config.batch_size
        loss = 0
        for _ in range(epochs):
            st = 0
            while st + batch_size < len(self.train_data):
                x = torch.tensor(self.train_data[st:st + batch_size][0]).to(torch.long)
                y = torch.tensor(self.train_data[st:st + batch_size][1]).to(torch.float32)
                loss, y_ = self.train_batch(x, y)
            if st + batch_size >= len(self.train_data):
                x = torch.tensor(self.train_data[st:][0]).to(torch.long)
                y = torch.tensor(self.train_data[st:][1]).to(torch.float32)
                loss, y_ = self.train_batch(x, y)
        return loss

    def evaluate(self):
        hit = 0
        ndcg = 0
        x = torch.tensor(self.test_data[0]).to(torch.long)
        gt_item = self.test_data[1]
        y_ = self.model(x)
        _, indices = torch.topk(y_, Config.top_k)
        indices = indices.tolist()
        if gt_item in indices:
            hit = 1
            ndcg = math.log(2) / np.log2(indices.index(gt_item) + 2)
        return hit, ndcg
