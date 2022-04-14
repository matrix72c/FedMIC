from time import time

import numpy as np

from client import Client
from config import Config
from server import Server


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
    Parses the training data: data[user_id] = [[[user_id, item_id_0], [user_id, item_id_1], ...], [rate0, rate1, ...]].
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
    mat = np.zeros((user_num, item_num))
    d = [[[], []] for _ in range(user_num)]
    for user_id, item_id, rate in data_list:
        d[user_id][0].append([user_id, item_id])
        d[user_id][1].append(rate)
        mat[user_id, item_id] = rate
    for user_id in range(user_num):
        for item_id in range(item_num):
            if mat[user_id, item_id] == 0:
                d[user_id][0].append([user_id, item_id])
                d[user_id][1].append(0)
    t2 = time()
    print("Parsed training data time: [%.1f s]" % (t2 - t1))
    return d, user_num, item_num


def parse_test_data(lines):
    """
    Parses the test data and returns a matrix.
    """
    test_list = [[] for _ in range(len(lines))]
    ret_test = []
    for line in lines:
        tmp_str_list = line.split("\t")
        user_str, item_str = tmp_str_list[0][1:-1].split(",")
        item_list = [int(i) for i in tmp_str_list[1:]]
        user = int(user_str)
        gt_item = int(item_str)
        item_list.append(gt_item)
        test_list[user] = [item_list, [gt_item]]
    for i in range(len(test_list)):
        tmp = []
        for j in range(len(test_list[i][0])):
            tmp.append([i, test_list[i][0][j]])
        ret_test.append([tmp, test_list[i][1]])

    return ret_test


def get_clients(train_data, test_data, user_num, item_num):
    """
    Distribute the train data and test data to clients.
    """
    client_list = []
    for i in range(len(train_data)):
        c = Client(train_data[i], test_data[i], user_num, item_num)
        client_list.append(c)
    return client_list


def main():

    train_lines = load_file(Config.train_data_path)
    test_lines = load_file(Config.test_data_path)
    train_data, user_num, item_num = parse_train_data(train_lines)
    test_data = parse_test_data(test_lines)
    client_list = get_clients(train_data, test_data, user_num, item_num)
    server = Server(client_list, user_num, item_num)
    server.run()


if __name__ == "__main__":
    main()
