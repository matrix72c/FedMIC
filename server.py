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
    def __init__(self, client_list, user_num, item_num, test_data, logger):
        self.clients = client_list
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
        for client in clients:
            client.model.load_state_dict(self.model.state_dict())
            loss.append(client.train())
            client_batch, _ = client.get_distill_batch()
            if distill_batch is None:
                distill_batch = client_batch
            else:
                distill_batch = torch.cat((distill_batch, client_batch), dim=0)

        # get avg logits
        distill_data = Data.TensorDataset(distill_batch, torch.tensor([0 for _ in range(distill_batch.shape[0])]))
        distill_loader = Data.DataLoader(distill_data, batch_size=Config.distill_batch_size, shuffle=False)
        logits_list = []
        for client in clients:
            client_logits = []
            for batch, _ in distill_loader:
                batch = batch.to(Config.device)
                pred = client.model(batch)
                client_logits.extend(pred.detach().cpu().numpy())
            client_logits = torch.tensor(client_logits)
            logits_list.append(client_logits)
        avg_logits = sum(logits_list) / len(logits_list)
        distill_logits = avg_logits

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
            if rnd % Config.eval_every == 0:
                tqdm.write("Round: %d, Time: %.1fs, Loss: %.4f, distill_loss: %.4f, Hit: %.4f, NDCG: %.4f" % (
                    rnd, time.time() - t, loss, distill_loss, hit, ndcg))
                t = time.time()
