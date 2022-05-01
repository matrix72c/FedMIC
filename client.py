from torch import nn
import torch.utils.data as Data
from model import NCFModel
from utils import *
from Logger import log_client_loss
import random


class Client:
    def __init__(self, train_data, train_label, test_data, user_num, item_num, logger, client_id=0):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.user_num = user_num
        self.item_num = item_num
        self.logger = logger
        self.client_id = client_id
        self.fake_id_list = [random.randint(0, self.user_num - 1) for _ in range(Config.num_fake)]
        self.model = NCFModel(user_num, item_num).to(Config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.learning_rate)
        self.dataset = NCFDataset(torch.tensor(train_data).to(torch.long), torch.tensor(train_label).to(torch.float32))
        self.loader = Data.DataLoader(self.dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)

    def train_batch(self, x, y):
        y_ = self.model(x)
        loss = nn.BCEWithLogitsLoss()(y_, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), y_.detach()

    def train(self):
        self.model.train()
        epochs = Config.epochs
        loss = 0
        for epoch in range(epochs):
            batch_loss_list = []
            for data in self.loader:
                x = data[0].to(Config.device)
                y = data[1].to(Config.device)
                loss, y_ = self.train_batch(x, y)
                batch_loss_list.append(loss)
            self.schedule.step()
            self.logger.log_client_loss(self.client_id, epoch, np.mean(batch_loss_list).item())
        return loss

    def get_distill_batch(self):
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
        _, indices = torch.topk(total_logits, Config.distill_batch_size * Config.distill_pos_ratio)
        positive_data = total_data[indices]
        positive_logits = total_logits[indices]

        # get the rest of items
        total_data = torch_delete(total_data, indices)
        total_logits = torch_delete(total_logits, indices)

        # get neg items id and corresponding logits
        neg_samples = torch.randint(0, len(total_data), (int(Config.distill_batch_size * (1 - Config.distill_pos_ratio)),))
        negative_data = total_data[neg_samples]
        negative_logits = total_logits[neg_samples]

        # concat positive and negative samples
        client_batch = torch.cat([positive_data, negative_data], dim=0)
        client_logits = torch.cat([positive_logits, negative_logits], dim=0)

        # add obfuscation
        obfuscate_list = []
        obfuscate_size = len(client_batch)
        for fake_id in self.fake_id_list:
            obfuscate_list.extend([[fake_id, random.randint(0, self.item_num - 1)] for _ in range(obfuscate_size)])
        obfuscate_data = torch.tensor(obfuscate_list)
        obfuscate_logits = []
        obfuscate_dataset = NCFDataset(obfuscate_data, [1. for _ in range(len(obfuscate_list))])
        obfuscate_dataloader = Data.DataLoader(obfuscate_dataset, batch_size=Config.batch_size, shuffle=False)
        for data, label in obfuscate_dataloader:
            data = data.to(Config.device)
            pred = self.model(data)
            obfuscate_logits.extend(pred.detach().cpu().numpy())
        obfuscate_logits = torch.tensor(obfuscate_logits)
        client_batch = torch.cat([client_batch, obfuscate_data], dim=0)
        client_logits = torch.cat([client_logits, obfuscate_logits], dim=0)
        return client_batch, client_logits
