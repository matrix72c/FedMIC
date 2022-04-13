import copy
import torch
from tqdm import tqdm
from model import NCFModel
import math
import numpy as np


class Server:
    def __init__(self, client_list, user_num, item_num, test_data, latent_dim=32, rounds=200, device="cuda"):
        self.clients = client_list
        self.user_num = user_num
        self.item_num = item_num
        self.test_data = test_data
        self.latent_dim = latent_dim
        self.rounds = rounds
        self.server_model = NCFModel(user_num, item_num, predictive_factor=latent_dim)
        self.device = device

    def iterate(self, rnd=0):  # rnd -> round
        single_round_results = {key: [] for key in ["num_users", "loss"]}
        bar = tqdm(enumerate(self.clients), total=len(self.clients))
        models_dict = []
        for client_id, client in bar:
            results = client.train()
            for k, i in results.items():
                single_round_results[k].append(i)
            printing_single_round = {"round": rnd}
            printing_single_round.update({k: round(sum(i) / len(i), 4) for k, i in single_round_results.items()})
            models_dict.append(client.model.state_dict())
            bar.set_description(str(printing_single_round))
        bar.close()
        server_new_dict = copy.deepcopy(models_dict[0])
        for i in range(1, len(models_dict)):
            client_dict = models_dict[i]
            for k in client_dict.keys():
                server_new_dict[k] += client_dict[k]
        for k in server_new_dict.keys():
            server_new_dict[k] /= len(models_dict)
        self.server_model.load_state_dict(server_new_dict)

    def evaluate(self, top_k=10):
        hits = []
        ndcgs = []
        for test_batch, gt_item in self.test_data:
            test_batch_tensor = torch.tensor(test_batch).to(self.device)
            predictions = self.server_model(test_batch_tensor)
            _, indices = torch.topk(predictions, top_k)
            recommends = torch.take(
                test_batch, indices).cpu().numpy().tolist()
            gt_item = test_batch[-1].item()
            if gt_item in recommends:
                hits.append(1)
                ndcgs.append(math.log(2) / math.log(recommends.index(gt_item) + 2))
            else:
                hits.append(0)
                ndcgs.append(0)
        return np.array(hits).mean(), np.array(ndcgs).mean()

    def run(self):
        for rnd in range(self.rounds):  # rnd -> round
            _ = [client.model.to(self.device) for client in self.clients]
            _ = [client.model.load_server_weights(self.server_model) for client in self.clients]
            self.iterate(rnd)
            hr, ndcg = self.evaluate()
            print("Round: {}, HR: {}, NDCG: {}".format(rnd, hr, ndcg))
