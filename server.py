import copy
import random
from model import NCFModel
from utils import *


class Server:
    def __init__(self, client_list, user_num, item_num, test_data):
        self.clients = client_list
        self.user_num = user_num
        self.item_num = item_num
        self.test_data = test_data
        self.model = NCFModel(user_num, item_num).to(Config.device)

    def iterate(self, rnd=0):  # rnd -> round
        """
        train sample model and update model
        """
        clients = random.sample(self.clients, Config.sample_size)
        loss = []
        models_dict = []
        for client in clients:
            loss.append(client.train(rnd))
            models_dict.append(client.model.state_dict())

        # update model
        server_new_dict = copy.deepcopy(models_dict[0])
        for i in range(1, len(models_dict)):
            client_dict = models_dict[i]
            for k in client_dict.keys():
                server_new_dict[k] += client_dict[k]
        for k in server_new_dict.keys():
            server_new_dict[k] /= len(models_dict)
        self.model.load_state_dict(server_new_dict)

        # set model
        for client in self.clients:
            client.model.load_state_dict(self.model.state_dict())

        # evaluate model
        hit, ndcg = evaluate(self.model, self.test_data)
        print("Loss: %.4f, Hit: %.4f, NDCG: %.4f" % (np.mean(loss), hit, ndcg))

    def run(self):
        for rnd in range(Config.rounds):  # rnd -> round
            self.iterate(rnd)
