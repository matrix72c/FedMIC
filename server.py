import configparser
import copy
import random
import time

import torch
from tqdm import tqdm
from model import NCFModel
import torch.utils.data as Data
import numpy as np
from utils import *
from torch import nn
from Logger import log_distill_result
from torch.optim.lr_scheduler import StepLR


class Server:
    def __init__(self, client_list, train_data, user_num, item_num, test_data, logger):
        self.clients = client_list
        self.train_data = train_data
        self.user_num = user_num
        self.item_num = item_num
        self.test_data = test_data
        self.model = NCFModel(user_num, item_num).to(Config.device)
        self.distill_loss_func = nn.KLDivLoss(reduction='batchmean')
        self.distill_optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.distill_learning_rate)
        self.schedule = StepLR(self.distill_optimizer, step_size=Config.distill_lr_step, gamma=Config.distill_lr_decay)
        self.logger = logger

    def iterate(self, rnd=0):  # rnd -> round
        """
        Train sampled model and aggregate model by FedAvg.
        """
        self.model.train()
        t = time.time()
        clients = random.sample(self.clients, Config.sample_size)
        loss = []
        distill_loss = None
        distill_batch = None
        distill_logits = None
        for client in clients:
            client.model_state_dict = self.model.state_dict()
            loss.append(client.train())

        if Config.fed_method == "FedAvg" or Config.fed_method == "FedProx":
            models_dict = []
            for client in clients:
                models_dict.append(client.model.state_dict())
            server_new_dict = copy.deepcopy(models_dict[0])
            for i in range(1, len(models_dict)):
                client_dict = models_dict[i]
                for k in client_dict.keys():
                    server_new_dict[k] += client_dict[k]
            for k in server_new_dict.keys():
                server_new_dict[k] /= len(models_dict)
            self.model.load_state_dict(server_new_dict)
            return np.mean(loss).item(), 0

        # ONLY fed distill method reach here
        assert Config.fed_method != "FedAvg" and Config.fed_method != "FedProx"
        if Config.fed_method == "FedDD":
            for client in clients:
                client_batch, client_logits = client.get_distill_batch()
                if distill_batch is None:
                    distill_batch = client_batch
                    distill_logits = client_logits
                else:
                    distill_batch = torch.cat((distill_batch, client_batch), dim=0)
                    distill_logits = torch.cat((distill_logits, client_logits), dim=0)
        elif Config.fed_method == "rand-avg":
            random_batch = torch.tensor(random.sample(self.train_data, len(clients) * Config.distill_batch_size))
            total_dataset = NCFDataset(random_batch, [1. for _ in range(len(random_batch))])
            total_dataloader = Data.DataLoader(total_dataset, batch_size=Config.batch_size, shuffle=False)
            logits_list = []
            for client in clients:
                logits = []
                for data, label in total_dataloader:
                    data = data.to(Config.device)
                    pred = client.model(data)
                    logits.extend(pred.detach().cpu().numpy())
                logits_list.append(torch.tensor(logits))
            distill_batch = random_batch
            distill_logits = sum(logits_list) / len(logits_list)
        elif Config.fed_method == "rand-direct":
            random_batch = torch.tensor(random.sample(self.train_data, Config.distill_batch_size))
            total_dataset = NCFDataset(random_batch, [1. for _ in range(len(random_batch))])
            total_dataloader = Data.DataLoader(total_dataset, batch_size=Config.batch_size, shuffle=False)
            distill_batch = torch.tensor([], dtype=torch.int).to(Config.device)
            distill_logits = torch.tensor([]).to(Config.device)
            for client in clients:
                for data, label in total_dataloader:
                    data = data.to(Config.device)
                    pred = client.model(data)
                    distill_batch = torch.cat((distill_batch, data), 0)
                    distill_logits = torch.cat((distill_logits, pred.detach()), 0)
            distill_batch = distill_batch.cpu()
            distill_logits = distill_logits.cpu()
            pass
        else:
            print("Invalid Method: ", Config.fed_method, "!")
            exit(0)
        assert distill_batch is not None
        assert distill_logits is not None
        distill_data = Data.TensorDataset(distill_batch, distill_logits)
        distill_loader = Data.DataLoader(distill_data, batch_size=Config.batch_size, shuffle=True, drop_last=True)
        for _ in range(Config.distill_epochs):
            distill_batch_loss_list = []
            for batch, logits in distill_loader:
                batch = batch.to(Config.device)
                logits = logits.to(Config.device)

                self.distill_optimizer.zero_grad()
                predict = self.model(batch)
                logits_softmax = torch.softmax(logits / Config.distill_T, dim=0)
                predict_softmax = torch.softmax(predict / Config.distill_T, dim=0)
                no_zero = torch.where(predict_softmax == 0, predict_softmax + 1e-10, predict_softmax)
                batch_loss = self.distill_loss_func(no_zero.log(), logits_softmax)
                batch_loss.backward()
                self.distill_optimizer.step()
                distill_batch_loss_list.append(batch_loss.item())
            distill_loss = np.mean(distill_batch_loss_list)
            if distill_loss < Config.distill_loss_threshold:
                break
        self.schedule.step()
        return np.mean(loss).item(), distill_loss.item()

    def run(self):
        t = time.time()
        for rnd in range(Config.rounds):  # rnd -> round
            loss, distill_loss = self.iterate(rnd)
            hit, ndcg = evaluate(self.model, self.test_data)
            self.logger.log_distill_result(rnd, 0, hit, ndcg)
            # evaluate model
            if rnd < 10 or rnd % Config.eval_every == 0:
                tqdm.write("Round: %d, Time: %.1fs, Loss: %.4f, distill_loss: %.4f, Hit: %.4f, NDCG: %.4f" % (
                    rnd, time.time() - t, loss, distill_loss, hit, ndcg))
                t = time.time()
