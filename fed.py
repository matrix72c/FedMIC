import argparse
from server import Server
from utils import *
from client import Client
import time


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
    Config.distill_lr_step = user_num // Config.sample_size
    clients_train_data, clients_train_label = distribute_data(train_data, train_label, user_num)
    client_list = get_clients(clients_train_data, clients_train_label, test_data, user_num, item_num, logger)
    server = Server(client_list, train_data, user_num, item_num, test_data, logger)
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fed_method", type=str, default=None)  # FedDD rand-avg rand-direct FedAvg FedProx
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--result_path", type=str, default="./result/"
                                                           + time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
                                                           + '/')
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--distill_pos_ratio", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None)
    parsed_args = parser.parse_args()
    Config.fed_method = parsed_args.fed_method if parsed_args.fed_method is not None else Config.fed_method
    Config.distill_T = parsed_args.T if parsed_args.T is not None else Config.distill_T
    Config.sample_size = parsed_args.sample_size if parsed_args.sample_size is not None else Config.sample_size
    Config.result_path = parsed_args.result_path if parsed_args.result_path is not None else Config.result_path
    Config.epochs = parsed_args.epochs if parsed_args.epochs is not None else Config.epochs
    Config.model = parsed_args.model if parsed_args.model is not None else Config.model
    Config.distill_pos_ratio = parsed_args.distill_pos_ratio if parsed_args.distill_pos_ratio is not None \
        else Config.distill_pos_ratio
    if parsed_args.dataset is not None:
        Config.train_data_path = "data/" + parsed_args.dataset + ".train.rating"
        Config.test_data_path = "data/" + parsed_args.dataset + ".test.negative"
    else:
        Config.train_data_path = parsed_args.train_data if parsed_args.train_data is not None\
            else Config.train_data_path
        Config.test_data_path = parsed_args.test_data if parsed_args.test_data is not None\
            else Config.test_data_path
    main()
