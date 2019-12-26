import torch
import torch.nn as nn
import numpy as np
from belieftracker.UtteranceSequence import UtteranceSequence

class BeliefTracker(nn.Module):
    def __init__(self, output_dim, config):
        super(BeliefTracker, self).__init__()
        self.gpu = config.use_gpu
        self.hidden_dim = config.hidden_dim
        self.output_dim = output_dim
        self.utterance_encoder = UtteranceSequence(config)
        self.hid2out = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = nn.Softmax(-1)
        
        if self.gpu:
            self.hid2out = self.hid2out.cuda()
            self.softmax = self.softmax.cuda()

    def forward(self, word_inputs, word_seq_lengths, is_train):
        current_hidden_state = self.utterance_encoder(word_inputs, word_seq_lengths)
        current_output = self.hid2out(current_hidden_state)
        if not is_train:
            current_output = self.softmax(current_output)
        return current_output