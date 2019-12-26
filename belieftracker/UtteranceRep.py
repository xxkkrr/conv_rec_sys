import torch
import torch.nn as nn
import numpy as np

class UtteranceRep(nn.Module):
    def __init__(self, config):
        super(UtteranceRep, self).__init__()
        self.gpu = config.use_gpu
        self.embedding_dim = config.word_embedding_dim
        self.ngram_embedding = config.ngram_embedding
        self.drop = nn.Dropout(config.dp)
        self.word_alphabet_size = config.word_alphabet_size
        self.word_embedding = nn.Embedding(config.word_alphabet_size, self.embedding_dim)
        self.padding_word_idx = config.word_alphabet_size
        # if self.ngram_embedding:
        #     self.get_ngram_embedding = lambda x: nn.functional.one_hot(x, config.word_alphabet_size+1)
        if config.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(config.word_alphabet_size, self.embedding_dim)))
        if self.gpu:
            self.drop = self.drop.cuda()
            self.word_embedding = self.word_embedding.cuda()


    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def forward(self, word_inputs, word_seq_lengths):
        if self.ngram_embedding:
            # word_embs = self.get_ngram_embedding(word_inputs)
            word_embs = nn.functional.one_hot(word_inputs, self.word_alphabet_size+1)
            if self.gpu:
                word_embs = word_embs.cuda()
        else:
            word_embs = self.word_embedding(word_inputs)
        word_embs = torch.sum(word_embs, -2)
        if not self.ngram_embedding:
            word_represent = self.drop(word_embs)
        else:
            word_embs[..., self.padding_word_idx] = 0
            word_represent = word_embs.float()
        return word_represent