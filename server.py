import copy
import random
import time
from tqdm import tqdm
from model import NCFModel
from utils import *
from Logger import log_distill_result


class Server:
    def __init__(self, client_list, user_num, item_num, test_data, logger):
        self.clients = client_list
        self.user_num = user_num
        self.item_num = item_num
        self.test_data = test_data
        self.model = NCFModel(user_num, item_num).to(Config.device)
        self.logger = logger

    def iterate(self, rnd=0):  # rnd -> round
        """
        Train sampled model and aggregate model by FedAvg.
        """
        t = time.time()
        clients = random.sample(self.clients, Config.sample_size)
        loss = []
        models_dict = []
        # loop = tqdm(enumerate(clients), total=len(clients))
        # for i, client in enumerate(clients):
        for client in clients:
            client.model.load_state_dict(self.model.state_dict())
            loss.append(client.train())
            models_dict.append(client.model.state_dict())
            # loop.set_description("Round: %d, Client: %d" % (rnd, client.client_id))
            # loop.set_postfix(loss=loss[-1])

        # FedAvg
        server_new_dict = copy.deepcopy(models_dict[0])
        for i in range(1, len(models_dict)):
            client_dict = models_dict[i]
            for k in client_dict.keys():
                server_new_dict[k] += client_dict[k]
        for k in server_new_dict.keys():
            server_new_dict[k] /= len(models_dict)
        self.model.load_state_dict(server_new_dict)

        return np.mean(loss).item()

    def run(self):
        t = time.time()
        for rnd in range(Config.rounds):  # rnd -> round
            loss = self.iterate(rnd)

            hit, ndcg = evaluate(self.model, self.test_data)
            self.logger.log_distill_result(rnd, 0, hit, ndcg)
            # evaluate model
            if rnd % Config.eval_every == 0:
                tqdm.write("Round: %d, Time: %.1fs, Loss: %.4f, Hit: %.4f, NDCG: %.4f" % (
                    rnd, time.time() - t, loss, hit, ndcg))
                time.sleep(1)
                t = time.time()
