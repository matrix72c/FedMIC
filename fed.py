from asyncio.log import logger
from model import NCFModel
from server import Server
from utils import *
from client import Client
from Logger import init_logger


def get_clients(clients_train_data, clients_train_label, test_data, user_num, item_num, logger):
    """
    Distribute the train data and test data to clients.
    """
    client_list = []
    for i in range(len(clients_train_data)):
        c = Client(clients_train_data[i], clients_train_label[i], test_data, user_num, item_num, logger, i)
        client_list.append(c)
    return client_list


def main():
    """
    Construct the NCF model and run the federated learning.
    """
    logger = Logger()
    torch.backends.cudnn.benchmark = True
    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_num, item_num, train_data, train_label, test_data = get_ncf_data()
    clients_train_data, clients_train_label = distribute_data(train_data, train_label, user_num)
    client_list = get_clients(clients_train_data, clients_train_label, test_data, user_num, item_num, logger)
    server = Server(client_list, user_num, item_num, test_data, logger)
    server.run()


if __name__ == "__main__":
    main()
