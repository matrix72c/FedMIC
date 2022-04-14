import torch
from utils import *
from client import Client


def main():
    torch.backends.cudnn.benchmark = True
    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_num, item_num, train_data, train_label, test_data = get_ncf_data()
    client = Client(train_data, train_label, test_data, user_num, item_num)
    loss = client.train()


if __name__ == "__main__":
    main()
