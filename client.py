import numpy as np
import torch
import torch.nn.functional as F
from evaluate import compute_metrics


class DataLoader:
    def __init__(self, train_data, default=None, seed=0):
        self.positives = train_data[0]
        self.negatives = train_data[1]
        self.ui_matrix = train_data[2]
        if default is None:
            self.default = np.array([[0, 0]]), np.array([0])
        else:
            self.default = default

    def delete_indexes(self, indexes, arr="pos"):
        if arr == "pos":
            self.positives = np.delete(self.positives, indexes, 0)
        else:
            self.negatives = np.delete(self.negatives, indexes, 0)

    def get_batch(self, batch_size):
        if self.positives.shape[0] < batch_size // 4 or self.negatives.shape[0] < batch_size - batch_size // 4:
            return torch.tensor(self.default[0]), torch.tensor(self.default[1])
        try:
            pos_indexes = np.random.choice(self.positives.shape[0], batch_size // 4)
            neg_indexes = np.random.choice(self.negatives.shape[0], batch_size - batch_size // 4)
            pos = self.positives[pos_indexes]
            neg = self.negatives[neg_indexes]
            self.delete_indexes(pos_indexes, "pos")
            self.delete_indexes(neg_indexes, "neg")
            batch = np.concatenate((pos, neg), axis=0)
            if batch.shape[0] != batch_size:
                return torch.tensor(self.default[0]), torch.tensor(self.default[1]).float()
            np.random.shuffle(batch)
            y = np.array([self.ui_matrix[i][j] for i, j in batch])
            return torch.tensor(batch), torch.tensor(y).float()
        except:
            return torch.tensor(self.default[0]), torch.tensor(self.default[1]).float()


class Client:
    def __init__(self, train_data, test_data, gt_item, model, epochs=10, batch_size=128, learning_rate=5e-4,
                 device="cuda"):
        self.ui_matrix = train_data[2]
        self.test_data = test_data
        self.gt_item = gt_item
        self.model = model
        self.loader = DataLoader(train_data)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

    def train_batch(self, x, y, optimizer):
        y_ = self.model(x)
        mask = (y > 0).float()
        loss = torch.nn.functional.mse_loss(y_ * mask, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), y_.detach()

    def train(self, optimizer, epochs=None, return_progress=False):
        self.model.join_output_weights()
        epoch = 0
        progress = {"epoch": [], "loss": [], "hit_ratio@10": [], "ndcg@10": []}
        running_loss, running_hr, running_ndcg = 0, 0, 0
        prev_running_loss, prev_running_hr, prev_running_ndcg = 0, 0, 0
        results = {}
        if epochs is None:
            epochs = self.epochs
        steps, prev_steps, prev_epoch = 0, 0, 0
        while epoch < epochs:
            x, y = self.loader.get_batch(self.batch_size)
            if x.shape[0] < self.batch_size:
                prev_running_loss, prev_running_hr, prev_running_ndcg = running_loss, running_hr, running_ndcg
                running_loss = 0
                running_hr = 0
                running_ndcg = 0
                prev_steps = steps
                steps = 0
                epoch += 1
                x, y = self.loader.get_batch(self.batch_size)
            x, y = x.int(), y.float()
            x, y = x.to(self.device), y.to(self.device)
            loss, y_ = self.train_batch(x, y, optimizer)
            hr, ndcg = compute_metrics(y.cpu().numpy(), y_.cpu().numpy())
            running_loss += loss
            running_hr += hr
            running_ndcg += ndcg
            if epoch != 0 and steps == 0:
                results = {"epoch": prev_epoch, "loss": prev_running_loss / (prev_steps + 1),
                           "hit_ratio@10": prev_running_hr / (prev_steps + 1),
                           "ndcg@10": prev_running_ndcg / (prev_steps + 1)}
            else:
                results = {"epoch": prev_epoch, "loss": running_loss / (steps + 1),
                           "hit_ratio@10": running_hr / (steps + 1), "ndcg@10": running_ndcg / (steps + 1)}
            steps += 1
            if prev_epoch != epoch:
                progress["epoch"].append(results["epoch"])
                progress["loss"].append(results["loss"])
                progress["hit_ratio@10"].append(results["hit_ratio@10"])
                progress["ndcg@10"].append(results["ndcg@10"])
                prev_epoch += 1
        r_results = {"num_users": self.ui_matrix.shape[0]}
        r_results.update({i: results[i] for i in ["loss", "hit_ratio@10", "ndcg@10"]})
        if return_progress:
            return r_results, progress
        else:
            return r_results
