import copy
import random
import time
from tqdm import tqdm
from model import NCFModel
import torch.utils.data as Data
import numpy as np
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
        self.model.train()
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

            # predict all items
            client_model.eval()
            total_data = torch.tensor([[client.client_id, i] for i in range(self.item_num)])
            total_logits = []
            total_dataset = NCFDataset(total_data, [1. for _ in range(self.item_num)])
            total_dataloader = Data.DataLoader(total_dataset, batch_size=Config.batch_size, shuffle=False)
            for data, label in total_dataloader:
                data = data.to(Config.device)
                pred = client_model(data)
                total_logits.extend(pred.detach().cpu().numpy())
            total_logits = torch.tensor(total_logits)

            # get positive items (batch size // 5)
            _, indices = torch.topk(total_logits, Config.distill_batch_size // 5 * 2)
            # _, indices = torch.topk(total_logits, Config.distill_batch_size)
            positive_data = total_data[indices]
            positive_logits = total_logits[indices]

            # get the rest of items
            total_data = torch_delete(total_data, indices)
            total_logits = torch_delete(total_logits, indices)
            for _ in range(Config.distill_epochs):
                # get neg items id and corresponding logits
                neg_samples = torch.randint(0, len(total_data), (Config.distill_batch_size // 5 * 3
                                                                 ,))
                negative_data = total_data[neg_samples]
                negative_logits = total_logits[neg_samples]

                # concat positive and negative samples
                client_batch = torch.cat([positive_data, negative_data], dim=0).to(Config.device)
                client_logits = torch.cat([positive_logits, negative_logits], dim=0).to(Config.device)
                # client_batch = positive_data.to(Config.device)
                # client_logits = positive_logits.to(Config.device)

                # start real distill epoch
                client_softmax = torch.softmax(client_logits, dim=0)
                # client_logits = client_model(data_batch)
                server_logits = self.model(client_batch)
                server_softmax = torch.softmax(server_logits, dim=0)
                distill_loss = nn.KLDivLoss()(server_softmax.log(), client_softmax)
                # distill_loss = nn.KLDivLoss()(server_logits, client_logits)
                distill_loss.backward()
                self.distill_optimizer.step()
                self.distill_optimizer.zero_grad()
        return np.mean(loss).item(), distill_loss.item()

    def run(self):
        t = time.time()
        for rnd in range(Config.rounds):  # rnd -> round
            loss, distill_loss = self.iterate(rnd)

            # evaluate model
            if rnd % Config.eval_every == 0:
                hit, ndcg = evaluate(self.model, self.test_data)
                tqdm.write("Round: %d, Time: %.1fs, Loss: %.4f, distill_loss: %.4f, Hit: %.4f, NDCG: %.4f" % (
                    rnd, time.time() - t, loss, distill_loss, hit, ndcg))
                t = time.time()
