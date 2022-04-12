from time import time
import numpy as np

from client import Client
from model import NCFModel
from server import Server

config = {
    "predictive_factor"
}


def load_file(file_path):
    """
    Loads a file and returns the contents as a string.
    """
    with open(file_path, "r") as f:
        t1 = time()
        lines = f.readlines()
        t2 = time()
        print("Read file %s time: [%.1f s]" % (file_path, t2 - t1))
        return lines


def parse_train_data(lines):
    """
    Parses the training data and returns a matrix.
    """
    t1 = time()
    data_list = []
    user_num = 0
    item_num = 0
    for line in lines:
        tmp_list = line.split("\t")
        user_id = int(tmp_list[0])
        item_id = int(tmp_list[1])
        user_num = max(user_id + 1, user_num)
        item_num = max(item_id + 1, item_num)
        rate = int(tmp_list[2])
        data_list.append([user_id, item_id, rate])
    ui_matrix = np.zeros((user_num, item_num))
    for data in data_list:
        ui_matrix[data[0], data[1]] = data[2]
    t2 = time()
    print("Parsed training data time: [%.1f s]" % (t2 - t1))
    return ui_matrix, user_num, item_num


def parse_test_data(lines):
    """
    Parses the test data and returns a matrix.
    """
    test_list = [[] for _ in range(len(lines))]
    for line in lines:
        tmp_str_list = line.split("\t")
        user_str, item_str = tmp_str_list[0][1:-1].split(",")
        item_list = [int(i) for i in tmp_str_list[1:]]
        user = int(user_str)
        gt_item = int(item_str)
        item_list.append(gt_item)
        test_list[user] = [item_list, gt_item]
    return test_list


def split_train_data(ui_matrix, user_num, item_num):
    """
    Split the train data.
    """
    t1 = time()
    clients_train_data = []
    for i in range(user_num):
        client_data = ui_matrix[i: i + 1]
        postives = np.argwhere(client_data > 0)
        negatives = np.argwhere(client_data == 0)
        clients_train_data.append([postives, negatives, ui_matrix[i:i + 1]])
    t2 = time()
    print("Distribute the train data time: [%.1f s]" % (t2 - t1))
    return clients_train_data


def get_clients(clients_train_data, clients_test_data, user_num, item_num):
    """
    Distribute the train data and test data to clients.
    """
    client_list = []
    for i in range(len(clients_train_data)):
        model = NCFModel(user_num, item_num, predictive_factor=32)
        c = Client(clients_train_data[i], clients_test_data[i][0], clients_test_data[i][1], model, epochs=10, batch_size=128, learning_rate=5e-4)
        client_list.append(c)
    return client_list


def main():
    train_data_path = "data/ml-100k.rating"
    test_data_path = "data/ml-100k.test.negative"
    train_lines = load_file(train_data_path)
    test_lines = load_file(test_data_path)
    train_data, user_num, item_num = parse_train_data(train_lines)
    clients_train_data = split_train_data(train_data, user_num, item_num)
    clients_test_data = parse_test_data(test_lines)
    client_list = get_clients(clients_train_data, clients_test_data, user_num, item_num)
    server = Server(client_list, user_num, item_num, latent_dim=32, rounds=200)
    server.run()


if __name__ == "__main__":
    main()
