import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayersModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gpu = config.use_gpu
        self.input_dim = config.input_dim
        self.hidden1_dim = config.hidden1_dim
        self.hidden2_dim = config.hidden2_dim
        self.linear1 = nn.Linear(self.input_dim, self.hidden1_dim)
        self.linear2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.drop = nn.Dropout(config.dp)

        if self.gpu:
            self.linear1 = self.linear1.cuda()
            self.linear2 = self.linear2.cuda()
            self.relu = self.relu.cuda()
            self.softmax = self.softmax.cuda()
            self.drop = self.drop.cuda()
        
    def forward(self, x, is_training):
        x = self.drop(x)
        hidden1 = self.relu(self.linear1(x))
        hidden1 = self.drop(hidden1)
        hidden2 = self.relu(self.linear2(hidden1))
        if is_training:
            out = hidden2
        else:
            out = self.softmax(hidden2)
        return out