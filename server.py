import copy
import random
import time
from tqdm import tqdm
from model import NCFModel
import torch.utils.data as Data
import numpy as np
from utils import *
from torch import nn
from Logger import log_distill_result


class Server:
    def __init__(self, client_list, user_num, item_num, test_data, logger):
        self.clients = client_list
        self.user_num = user_num
        self.item_num = item_num
        self.test_data = test_data
        self.model = NCFModel(user_num, item_num).to(Config.device)
        self.distill_loss_func = nn.KLDivLoss(reduction='batchmean')
        self.distill_optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.distill_learning_rate)
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
            client.model.load_state_dict(self.model.state_dict())
            loss.append(client.train())
            client_batch, client_logits = client.get_distill_batch()
            if distill_batch is None:
                distill_batch = client_batch
                distill_logits = client_logits
            else:
                distill_batch = torch.cat((distill_batch, client_batch), dim=0)
                distill_logits = torch.cat((distill_logits, client_logits), dim=0)

        distill_data = Data.TensorDataset(distill_batch, distill_logits)
        distill_loader = Data.DataLoader(distill_data, batch_size=Config.batch_size, shuffle=True)
        for _ in range(Config.distill_epochs):
            distill_batch_loss_list = []
            for batch, logits in distill_loader:
                batch = batch.to(Config.device)
                logits = logits.to(Config.device)

                self.distill_optimizer.zero_grad()
                predict = self.model(batch)
                logits_softmax = torch.softmax(logits / Config.distill_T, dim=0)
                predict_softmax = torch.softmax(predict / Config.distill_T, dim=0)
                batch_loss = self.distill_loss_func(predict_softmax.log(), logits_softmax)
                batch_loss.backward()
                self.distill_optimizer.step()
                distill_batch_loss_list.append(batch_loss.item())
            distill_loss = np.mean(distill_batch_loss_list)
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
