import copy
import random

import torch
from torch import nn
import torch.utils.data as Data
from model import NCFModel
from utils import *
from Logger import log_client_loss
from torch.optim.lr_scheduler import StepLR


class Client:
    def __init__(self, train_data, train_label, test_data, user_num, item_num, logger, client_id=0, mu=0):
        self.train_data = train_data
        self.pos_data = train_data[:(len(train_data) // (Config.neg_pos_ratio + 1))]
        self.neg_data = train_data[(len(train_data) // (Config.neg_pos_ratio + 1)):]
        self.train_label = train_label
        self.test_data = test_data
        self.user_num = user_num
        self.item_num = item_num
        self.logger = logger
        self.client_id = client_id
        self.mu = mu
        self.model_state_dict = None
        self.lr = Config.learning_rate
        self.lr_count = 0
        self.dataset = NCFDataset(torch.tensor(train_data).to(torch.long), torch.tensor(train_label).to(torch.float32))
        self.loader = Data.DataLoader(self.dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)

    def train(self):
        model = NCFModel(self.user_num, self.item_num).to(Config.device)
        assert self.model_state_dict is not None
        model.load_state_dict(self.model_state_dict)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        model.train()
        epochs = Config.epochs
        mean_loss = 0
        server_model = None
        if Config.fed_method == "FedProx":
            server_model = NCFModel(self.user_num, self.item_num).to(Config.device)
            server_model.load_state_dict(model.state_dict())
        for epoch in range(epochs):
            batch_loss_list = []
            for data in self.loader:
                x = data[0].to(Config.device)
                y = data[1].to(Config.device)
                y_ = model(x)
                if Config.fed_method == "FedProx":
                    proximal_term = 0.0
                    for w, w_t in zip(model.parameters(), server_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss = nn.BCEWithLogitsLoss()(y_, y) + (self.mu / 2) * proximal_term
                else:
                    loss = nn.BCEWithLogitsLoss()(y_, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_loss_list.append(loss.detach().item())
            mean_loss = np.mean(batch_loss_list)
            # lr schedule
            self.lr_count += 1
            if self.lr_count == Config.lr_step:
                self.lr_count = 0
                self.lr = self.lr * Config.lr_decay
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            # early stop
            if mean_loss < Config.local_loss_threshold:
                break
            self.logger.log_client_loss(self.client_id, epoch, np.mean(batch_loss_list).item())
        self.model_state_dict = model.state_dict()
        return mean_loss

    def get_distill_batch(self):
        model = NCFModel(self.user_num, self.item_num).to(Config.device)
        assert self.model_state_dict is not None
        model.load_state_dict(self.model_state_dict)

        num_positive = int(Config.distill_batch_size * Config.distill_pos_ratio)
        if num_positive < len(self.pos_data):
            positive_data = random.sample(self.pos_data, num_positive)
        else:
            positive_data = self.pos_data
        if Config.distill_batch_size - len(positive_data) < len(self.neg_data):
            negative_data = random.sample(self.neg_data, Config.distill_batch_size - len(positive_data))
        else:
            negative_data = self.neg_data
        positive_data.extend(negative_data)
        client_batch = torch.tensor(positive_data)

        dataset = NCFDataset(client_batch, [1. for _ in range(self.item_num)])
        dataloader = Data.DataLoader(dataset, batch_size=Config.batch_size, shuffle=False)
        logits = []
        for data, label in dataloader:
            data = data.to(Config.device)
            pred = model(data)
            logits.extend(pred.detach().cpu().numpy())
        client_logits = torch.tensor(logits)

        # # predict all items
        # total_data = torch.tensor([[self.client_id, i] for i in range(self.item_num)])
        # total_logits = []
        # total_dataset = NCFDataset(total_data, [1. for _ in range(self.item_num)])
        # total_dataloader = Data.DataLoader(total_dataset, batch_size=Config.batch_size, shuffle=False)
        # for data, label in total_dataloader:
        #     data = data.to(Config.device)
        #     pred = model(data)
        #     total_logits.extend(pred.detach().cpu().numpy())
        # total_logits = torch.tensor(total_logits)
        #
        # # get positive items
        # num_positive = int(Config.distill_batch_size * Config.distill_pos_ratio)
        # _, indices = torch.topk(total_logits, num_positive)
        # positive_data = total_data[indices]
        # positive_logits = total_logits[indices]
        #
        # # get the rest of items
        # total_data = torch_delete(total_data, indices)
        # total_logits = torch_delete(total_logits, indices)
        #
        # # get neg items id and corresponding logits
        # neg_samples = torch.randint(0, len(total_data), (Config.distill_batch_size - num_positive,))
        # negative_data = total_data[neg_samples]
        # negative_logits = total_logits[neg_samples]
        #
        # # concat positive and negative samples
        # client_batch = torch.cat([positive_data, negative_data], dim=0)
        # client_logits = torch.cat([positive_logits, negative_logits], dim=0)
        return client_batch, client_logits
