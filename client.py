import copy

from torch import nn
import torch.utils.data as Data
from model import NCFModel
from utils import *
from Logger import log_client_loss
from torch.optim.lr_scheduler import StepLR


class Client:
    def __init__(self, train_data, train_label, test_data, user_num, item_num, logger, client_id=0, mu=0):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.user_num = user_num
        self.item_num = item_num
        self.logger = logger
        self.client_id = client_id
        self.mu = mu
        self.model = NCFModel(user_num, item_num)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.learning_rate)
        self.schedule = StepLR(self.optimizer, step_size=Config.lr_step, gamma=Config.lr_decay)
        self.dataset = NCFDataset(torch.tensor(train_data).to(torch.long), torch.tensor(train_label).to(torch.float32))
        self.loader = Data.DataLoader(self.dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)

    def train_batch(self, x, y):
        y_ = self.model(x)
        loss = nn.BCEWithLogitsLoss()(y_, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), y_.detach()

    def prox_train(self, x, y, server_params):
        y_ = self.model(x)
        proximal_term = 0.0
        for w, w_t in zip(self.model.parameters(), server_params):
            proximal_term += (w - w_t).norm(2)
        loss = nn.BCEWithLogitsLoss()(y_, y) + (self.mu / 2) * proximal_term
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), y_.detach()

    def train(self):
        self.model = self.model.to(Config.device)
        self.model.train()
        epochs = Config.epochs
        loss = 0
        if Config.fed_method == "FedProx":
            server_model = copy.deepcopy(self.model)
            for epoch in range(epochs):
                batch_loss_list = []
                for data in self.loader:
                    x = data[0].to(Config.device)
                    y = data[1].to(Config.device)
                    loss, y_ = self.prox_train(x, y, server_model.parameters())
                    batch_loss_list.append(loss)
                mean_loss = np.mean(batch_loss_list)
                if mean_loss < Config.local_loss_threshold:
                    break
                self.schedule.step()
                self.logger.log_client_loss(self.client_id, epoch, np.mean(batch_loss_list).item())
        else:
            for epoch in range(epochs):
                batch_loss_list = []
                for data in self.loader:
                    x = data[0].to(Config.device)
                    y = data[1].to(Config.device)
                    loss, y_ = self.train_batch(x, y)
                    batch_loss_list.append(loss)
                mean_loss = np.mean(batch_loss_list)
                if mean_loss < Config.local_loss_threshold:
                    break
                self.schedule.step()
                self.logger.log_client_loss(self.client_id, epoch, np.mean(batch_loss_list).item())
        self.model = self.model.cpu()
        return loss

    def get_distill_batch(self):
        self.model = self.model.to(Config.device)
        self.model.eval()
        # predict all items
        total_data = torch.tensor([[self.client_id, i] for i in range(self.item_num)])
        total_logits = []
        total_dataset = NCFDataset(total_data, [1. for _ in range(self.item_num)])
        total_dataloader = Data.DataLoader(total_dataset, batch_size=Config.batch_size, shuffle=False)
        for data, label in total_dataloader:
            data = data.to(Config.device)
            pred = self.model(data)
            total_logits.extend(pred.detach().cpu().numpy())
        total_logits = torch.tensor(total_logits)

        # get positive items
        num_positive = int(Config.distill_batch_size * Config.distill_pos_ratio)
        _, indices = torch.topk(total_logits, num_positive)
        positive_data = total_data[indices]
        positive_logits = total_logits[indices]

        # get the rest of items
        total_data = torch_delete(total_data, indices)
        total_logits = torch_delete(total_logits, indices)

        # get neg items id and corresponding logits
        neg_samples = torch.randint(0, len(total_data), (Config.distill_batch_size - num_positive,))
        negative_data = total_data[neg_samples]
        negative_logits = total_logits[neg_samples]

        # concat positive and negative samples
        client_batch = torch.cat([positive_data, negative_data], dim=0)
        client_logits = torch.cat([positive_logits, negative_logits], dim=0)
        self.model = self.model.cpu()
        return client_batch, client_logits
