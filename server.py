import copy
import random
import time
from tqdm import tqdm
from model import NCFModel
from utils import *
from torch import nn


class Server:
    def __init__(self, client_list, user_num, item_num, test_data):
        self.clients = client_list
        self.user_num = user_num
        self.item_num = item_num
        self.test_data = test_data
        self.model = NCFModel(user_num, item_num).to(Config.device)
        self.distill_optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.distill_learning_rate)

    def iterate(self, rnd=0):  # rnd -> round
        """
        Train sampled model and aggregate model by FedAvg.
        """
        t = time.time()
        clients = random.sample(self.clients, Config.sample_size)
        loss = []
        # loop = tqdm(enumerate(clients), total=len(clients))
        # for i, client in enumerate(clients):
        for client in clients:
            client.model.load_state_dict(self.model.state_dict())
            loss.append(client.train())
            # loop.set_description("Round: %d, Client: %d" % (rnd, client.client_id))
            # loop.set_postfix(loss=loss[-1])

        for client in clients:
            client_dict = client.model.state_dict()
            client_model = NCFModel(self.user_num, self.item_num).to(Config.device)
            client_model.load_state_dict(client_dict)
            for _ in range(Config.distill_epochs):
                data_batch = torch.tensor([[client.client_id, random.randint(0, self.item_num - 1)]
                                           for _ in range(Config.distill_batch_size)]).to(Config.device)
                client_logits, client_softmax = client_model(data_batch, softmax=True)
                server_logits, server_softmax = self.model(data_batch, softmax=True)
                distill_loss = nn.KLDivLoss()(server_softmax, client_softmax)
                distill_loss.backward()
                self.distill_optimizer.step()
                self.distill_optimizer.zero_grad()

        return np.mean(loss).item()

    def run(self):
        t = time.time()
        for rnd in range(Config.rounds):  # rnd -> round
            loss = self.iterate(rnd)

            # evaluate model
            if rnd % Config.eval_every == 0:
                hit, ndcg = evaluate(self.model, self.test_data)
                tqdm.write("Round: %d, Time: %.1fs, Loss: %.4f, Hit: %.4f, NDCG: %.4f" % (
                    rnd, time.time() - t, loss, hit, ndcg))
                time.sleep(1)
                t = time.time()
