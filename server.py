import copy
import random
import time
from tqdm import tqdm
from model import NCFModel
from utils import *
import torch.nn.functional as F
import torch.nn as nn
from Logger import log_distill_result


class Server:
    def __init__(self, client_list, total_unlabeled_data, user_num, item_num, test_data, logger):
        self.clients = client_list
        self.total_unlabeled_data = total_unlabeled_data
        self.user_num = user_num
        self.item_num = item_num
        self.test_data = test_data
        self.logger = logger
        self.model = NCFModel(user_num, item_num).to(Config.device)
        self.distill_loss = nn.KLDivLoss(reduction='batchmean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.distill_learning_rate)

    def iterate(self, rnd=0):  # rnd -> round
        """
        Train sampled model and aggregate model by FedAvg.
        """
        t = time.time()
        clients = random.sample(self.clients, Config.sample_size)
        loss_list = []
        for client in clients:
            client.model.load_state_dict(self.model.state_dict())
            loss_list.append(client.train())
        distill_batch = random.sample(self.total_unlabeled_data, Config.distill_batch_size)
        distill_batch = torch.tensor(distill_batch).to(Config.device)

        for _ in range(Config.distill_epochs):
            # distill_batch = torch.tensor([[random.randint(0, self.user_num - 1), 
                                        #    random.randint(0, self.item_num - 1)] 
                                        #    for _ in  range(Config.distill_batch_size)]
                                        #    ).to(Config.device)
            logits_list = []
            for client in clients:
                logits_list.append(client.model(distill_batch))
            client_logits = sum(logits_list) / len(logits_list)
            client_softmax = F.softmax(client_logits / 3, dim=0)
            prediction_logits = self.model(distill_batch)
            prediction_softmax = F.softmax(prediction_logits / 3, dim=0)
            self.optimizer.zero_grad()
            loss = self.distill_loss(prediction_softmax.log(), client_softmax)
            loss.backward()
            self.optimizer.step()
        return np.mean(loss_list).item(), loss.item()

    def run(self):
        t = time.time()
        for rnd in range(Config.rounds):  # rnd -> round
            loss, distill_loss = self.iterate(rnd)

            hit, ndcg = evaluate(self.model, self.test_data)
            self.logger.log_distill_result(rnd, distill_loss, hit, ndcg)
            # evaluate model
            if rnd % Config.eval_every == 0:
                hit, ndcg = evaluate(self.model, self.test_data)
                tqdm.write("Round: %d, Time: %.1fs, Loss: %.4f, distill_loss: %.4f, Hit: %.4f, NDCG: %.4f" % (
                    rnd, time.time() - t, loss, distill_loss, hit, ndcg))
                time.sleep(1)
                t = time.time()
