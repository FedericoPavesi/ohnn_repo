import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import gc
"""
Object used to train the model for binary classification
"""

class TrainDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, indices):
        return self.x_data[indices], self.y_data[indices]




class modelTrainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history = {'train_loss' : [],
                        'train_accuracy' : [],
                        'validation_loss' : [],
                        'validation_accuracy' : [],
                        'f1_class0' : [],
                        'f1_class1' : []}
        self.state_dict = []
        self.best_configuration = None
        self.observer_history = {}
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.observer_history[name] = []
        self.observer_history['rule_weight'] = []
        # because observer history will store both forward and backward weights,
        # we split them into forward and backward
        self.forward_history = {}
        self.backward_history = {}
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.forward_history[name] = []
            self.backward_history[name] = []
        self.forward_history['rule_weight'] = []
        self.backward_history['rule_weight'] = []

        self.grad_history = {}
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.grad_history[name] = []
        self.grad_history['rule_weight'] = []

    def observer(self): #keeps track of weights evolution
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.observer_history[name].append(self.model.hidden_rules[rule].linear.weight.detach().clone())
        self.observer_history['rule_weight'].append(self.model.rule_weight.linear.weight.detach().clone())

    def grad_observer(self):
        for rule in range(self.model.num_rules):
            name = 'rule_' + str(rule)
            self.grad_history[name].append(self.model.hidden_rules[rule].linear.weight.grad.detach().clone())
        self.grad_history['rule_weight'].append(self.model.rule_weight.linear.weight.grad.detach().clone())


    def train_model(self, train_data_loader, num_epochs, device, learning_rate, val_data_loader=None,
                    store_weights_history=True, store_weights_grad=True):

        if device != 'cpu':
            gc.collect()
            torch.cuda.empty_cache()

        self.optimizer = self.optimizer(self.model.parameters(), lr = learning_rate)

        min_loss = 1e5

        for epoch in range(num_epochs):
            self.model.train()

            total_loss = 0
            correct = 0
            total = 0

            for x_train, y_train in (train_iterations := tqdm(train_data_loader)):

                self.model.to(device)
                x_train.to(device)
                y_train.to(device)

                x_train, y_train = x_train.float(), y_train.float()

                self.optimizer.zero_grad()

                batch_total = len(y_train)
                total += batch_total

                outputs = self.model(x_train).squeeze()

                predicted = torch.round(outputs)

                batch_correct = (predicted == y_train).sum().item()
                correct += batch_correct

                loss = self.loss_fn(outputs, y_train)

                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                f1 = f1_score(predicted.to('cpu').detach().numpy(),
                                        y_train.to('cpu').detach().numpy(),
                                        average = None,
                                        zero_division = 0,
                                       labels = np.array([0, 1]))

                train_iterations.set_description('batch_loss: {:.4f}, batch_accuracy: {:.4f}, f1 score 0: {:.4f}, f1 score 1: {:.4f}'.format(loss.item()/batch_total, batch_correct/batch_total, f1[0], f1[1]))

            avg_loss = total_loss/total
            accuracy = correct/total

            if val_data_loader is not None:
                val_loss, val_accuracy, val_f1_0, val_f1_1 = self.eval_model(val_data=val_data_loader, device=device)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                self.history['val_f1_0'].append(val_f1_0)
                self.history['val_f1_1'].append(val_f1_1)
            else:
                val_loss, val_accuracy, val_f1_0, val_f1_1 = None, None, None, None

            self.history['train_loss'].append(avg_loss)
            self.history['train_accuracy'].append(accuracy)

            if store_weights_history:
                self.observer() #<-- store weight history

            if store_weights_grad:
                self.grad_observer() # <-- stores weights gradient

            if val_data_loader is not None:
                if val_loss < min_loss:
                    min_loss = val_loss
                    save_model = self.model.clone()
                    self.best_configuration = [save_model.to('cpu').state_dict(), val_loss, epoch]


    def eval_model(self, val_data, device):
        if device != 'cpu':
            gc.collect()
            torch.cuda.empty_cache()

        self.model.to(device)
        self.model.eval()

        eval_loss = 0
        total_correct = 0
        total = len(val_data)
        f1_0 = 0
        f1_1 = 0
        num_iter = 0

        with torch.no_grad():
            for val_x, val_y in tqdm(val_data):

                val_x.to(device)
                val_y.to(device)

                outputs = self.model(val_x).squeeze()

                batch_pred = torch.round(outputs)

                loss = self.loss_fn(outputs, val_y)

                eval_loss = loss.item()

                batch_correct = (batch_pred == val_y).sum().item()
                total_correct += batch_correct

                f1 = f1_score(batch_pred.to('cpu').detach().numpy(),
                                val_y.to('cpu').detach().numpy(),
                                average = None,
                                zero_division = 0,
                               labels = np.array([0, 1]))

                f1_0 += f1[0]
                f1_1 += f1[1]
                num_iter += 1

        eval_loss = eval_loss/total
        accuracy = total_correct/total
        f1_0 /= num_iter
        f1_1 /= num_iter

        return eval_loss, accuracy, f1_0, f1_1

    def predict(self, val_data, device, prob=True):
        if device != 'cpu':
            gc.collect()
            torch.cuda.empty_cache()

        self.model.to(device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for val_x, val_y in tqdm(val_data):
                val_x.to(device)
                val_y.to(device)
                outputs = self.model(val_x).squeeze()
                if not prob:
                    outputs = torch.round(outputs)
                predictions.append(outputs.to('cpu'))
        predictions = torch.cat(predictions, axis=0)
        return predictions






