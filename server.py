import copy
import torch
import random
from tqdm import tqdm

from config import Config
from model import NCFModel
import math
import numpy as np


class Server:
    def __init__(self, client_list, user_num, item_num):
        self.clients = client_list
        self.user_num = user_num
        self.item_num = item_num
        self.server_model = NCFModel(user_num, item_num)

    def iterate(self, rnd=0):  # rnd -> round
        """
        train sample model and update model
        """
        clients = random.sample(self.clients, Config.sample_size)
        loss = 0
        models_dict = []
        for client in clients:
            loss += client.train(rnd)
            models_dict.append(client.model.state_dict())
        loss /= len(clients)

        # update model
        server_new_dict = copy.deepcopy(models_dict[0])
        for i in range(1, len(models_dict)):
            client_dict = models_dict[i]
            for k in client_dict.keys():
                server_new_dict[k] += client_dict[k]
        for k in server_new_dict.keys():
            server_new_dict[k] /= len(models_dict)
        self.server_model.load_state_dict(server_new_dict)

        # set model and evaluate model
        hits, ndcgs = [], []
        for client in self.clients:
            client.model.load_server_weights(self.server_model)
            hit, ndcg = client.evaluate()
            hits.append(hit)
            ndcgs.append(ndcg)
        hit, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print("Round: {}, Loss: {}, HR@10: {}, NDCG@10: {}".format(rnd, loss, hit, ndcg))

    def run(self):
        for rnd in range(Config.rounds):  # rnd -> round
            self.iterate(rnd)
