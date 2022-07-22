from utils import *
from client import Client
from model import NCFModel
import time


def main():
    dataset = "ml-100k"
    Config.train_data_path = "data/" + dataset + ".train.rating"
    Config.test_data_path = "data/" + dataset + ".test.negative"
    Config.result_path = "./result/" + time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time())) + '/'
    torch.backends.cudnn.benchmark = True
    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_num, item_num, train_data, train_label, test_data = get_ncf_data()
    logger = Logger()
    client = Client(train_data, train_label, test_data, user_num, item_num, logger)
    model = NCFModel(user_num, item_num)
    client.model_state_dict = model.state_dict()
    for i in range(20):
        client.train()
        model.load_state_dict(client.model_state_dict)
        hit, ndcg = evaluate(model, test_data)
        print("Iteration {}: hit: {}, ndcg: {}".format(i, hit, ndcg))


if __name__ == "__main__":
    main()
