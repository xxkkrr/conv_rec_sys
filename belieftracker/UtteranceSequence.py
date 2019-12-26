import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from belieftracker.UtteranceRep import UtteranceRep

class UtteranceSequence(nn.Module):
    def __init__(self, config):
        super(UtteranceSequence, self).__init__()
        self.gpu = config.use_gpu
        self.droplstm = nn.Dropout(config.dp)
        self.bilstm_flag = config.bilstm
        self.lstm_layer = config.layer_num
        self.batch_size = config.batch_size
        self.wordrep = UtteranceRep(config)
        if config.ngram_embedding:
            self.input_size = config.word_alphabet_size + 1
        else:
            self.input_size = config.word_embedding_dim
        if self.bilstm_flag:
            self.lstm_hidden = config.hidden_dim // 2
        else:
            self.lstm_hidden = config.hidden_dim

        self.word_feature_extractor = config.word_feature_extractor
        if self.word_feature_extractor == "GRU":
            self.lstm = nn.GRU(self.input_size, self.lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "LSTM":
            self.lstm = nn.LSTM(self.input_size, self.lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag, dropout=config.dp)

        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            self.lstm = self.lstm.cuda()


    def forward(self, word_inputs, word_seq_lengths, return_all=False):
        word_seq_lengths, perm_idx = word_seq_lengths.sort(0, descending=True)
        word_inputs = word_inputs[perm_idx]
        word_represent = self.wordrep(word_inputs, word_seq_lengths)
        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        lstm_out, (ht, ct) = self.lstm(packed_words)
        lstm_out, _ = pad_packed_sequence(lstm_out, True)
        _, unperm_idx = perm_idx.sort(0)
        if return_all:
            return self.droplstm(lstm_out[unperm_idx])
        else:
            if self.bilstm_flag:
                ht = ht.view(self.lstm_layer, 2, len(word_seq_lengths), self.lstm_hidden)
                htht = torch.cat([ht[-1][0],ht[-1][1]], -1)
                return self.droplstm(htht[unperm_idx])
            else:
                return self.droplstm(ht[-1][unperm_idx])