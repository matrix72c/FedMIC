from torch import nn
import torch.utils.data as Data
from model import NCFModel
from utils import *
from Logger import log_client_loss
from torch.optim.lr_scheduler import StepLR


class Client:
    def __init__(self, train_data, train_label, test_data, user_num, item_num, logger, client_id=0):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.user_num = user_num
        self.item_num = item_num
        self.logger = logger
        self.client_id = client_id
        self.mu = Config.mu
        self.model = NCFModel(user_num, item_num)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.learning_rate)
        self.schedule = StepLR(self.optimizer, step_size=Config.lr_step, gamma=Config.lr_decay)
        self.dataset = NCFDataset(torch.tensor(train_data).to(torch.long), torch.tensor(train_label).to(torch.float32))
        self.loader = Data.DataLoader(self.dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)

    def train_batch(self, x, y, server_params):
        y_ = self.model(x)
        proximal_term = 0.0
        for w, w_t in zip(self.model.parameters(), server_params):
            proximal_term += (w - w_t).norm(2)
        loss = nn.BCEWithLogitsLoss()(y_, y) + (self.mu / 2) * proximal_term
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), y_.detach()

    def train(self, server_state_dict):
        self.model = self.model.to(Config.device)
        self.model.load_state_dict(server_state_dict)
        server_model = NCFModel(self.user_num, self.item_num).to(Config.device)
        server_model.load_state_dict(server_state_dict)
        server_params = server_model.parameters()
        self.model.train()
        epochs = Config.epochs
        loss = 0
        for epoch in range(epochs):
            batch_loss_list = []
            for data in self.loader:
                x = data[0].to(Config.device)
                y = data[1].to(Config.device)
                loss, y_ = self.train_batch(x, y, server_params)
                batch_loss_list.append(loss)
            self.schedule.step()
            self.logger.log_client_loss(self.client_id, epoch, np.mean(batch_loss_list).item())
        self.model = self.model.to("cpu")
        return loss
