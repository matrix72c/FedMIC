import copy
import torch
from tqdm import tqdm
from model import ServerNeuralCollaborativeFiltering


class Server:
    def __init__(self, client_list, user_num, item_num, latent_dim, rounds, device="cuda"):
        self.clients = client_list
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.rounds = rounds
        self.server_model = ServerNeuralCollaborativeFiltering(item_num, predictive_factor=latent_dim)
        self.optimizers = [torch.optim.Adam(client.model.parameters(), lr=5e-4) for client in self.clients]
        self.device = device

    def iterate(self, rnd=0):  # rnd -> round
        single_round_results = {key: [] for key in ["num_users", "loss", "hit_ratio@10", "ndcg@10"]}
        bar = tqdm(enumerate(self.clients), total=len(self.clients))
        models_dict = []
        for client_id, client in bar:
            results = client.train(self.optimizers[client_id])
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
        server_new_dict.pop("mlp_user_embeddings.weight")
        server_new_dict.pop("gmf_user_embeddings.weight")
        self.server_model.load_state_dict(server_new_dict)

    def run(self):
        for rnd in range(self.rounds):  # rnd -> round
            _ = [client.model.to(self.device) for client in self.clients]
            _ = [client.model.load_server_weights(self.server_model) for client in self.clients]
            self.iterate(rnd)
