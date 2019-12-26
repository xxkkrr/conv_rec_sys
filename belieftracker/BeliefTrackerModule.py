import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import datetime
from nltk.util import ngrams
import re
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from belieftracker.BeliefTracker import BeliefTracker

def generate_ngrams(s, n):
    '''
    s: document, str type
    n: n-gram
    output: list of n-grams 
    '''
    tokens = [token for token in s.split(" ") if token != ""]
    output = []
    for n_ in range(1,n):
            output.extend(list(ngrams(tokens, n_)))
    return output

class BatchData():
    def __init__(self, sen_list, label_list, padding_word_idx, use_gpu, output_check=False):
        batch_size = len(sen_list)
        word_seq_lengths = [len(_) for _ in sen_list]
        max_word_seq_lengths = max(word_seq_lengths)
        max_word_num = max([len(item) for sublist in sen_list for item in sublist])
        self.words_inputs = np.zeros([batch_size, max_word_seq_lengths, max_word_num], dtype=np.int64) + padding_word_idx
        for index_x, sublist in enumerate(sen_list):
            for index_y, item in enumerate(sublist):
                item_len = len(item)
                self.words_inputs[index_x, index_y, :item_len] = item
        self.words_inputs = torch.tensor(self.words_inputs)
        self.word_seq_lengths = torch.tensor(word_seq_lengths)
        if label_list != None:
            self.label_list = torch.tensor(label_list)
        if use_gpu:
            self.words_inputs = self.words_inputs.cuda()
            self.word_seq_lengths = self.word_seq_lengths.cuda()
            if label_list != None:
                self.label_list = self.label_list.cuda()
        if output_check:
            self.output()

    def output(self):
        print("--------------------------")
        print("words_inputs:", self.words_inputs)
        print("word_seq_lengths:", self.word_seq_lengths)
        print("label_list:", self.label_list)

class BeliefTrackerModule():
    def __init__(self, config, load_dataset):
        self.belieftracker_model_path = config.belieftracker_model_path
        self.belieftracker_model_name_list = config.belieftracker_model_name_list
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.value_num_list = config.value_num_list
        self.index2facet = config.index2facet
        self.tracker_hidden_dim = config.hidden_dim
        self.padding_word_idx = config.word_alphabet_size
        self.belieftracker_list = [BeliefTracker(output_dim, config) \
                                    for output_dim in self.value_num_list]
        self.use_gpu = config.use_gpu
        self.epoch_num = config.epoch_num
        self.save_step = config.save_step
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.optimizer_list =[optim.SGD(belieftracker.parameters(), lr=config.lr, weight_decay=config.weight_decay) \
                                for belieftracker in self.belieftracker_list]
        self.train_log_path = config.train_log_path
        self.criterion = nn.CrossEntropyLoss()

        self.facet_all_data = None
        self.train_data_list = None
        self.eva_data_list = None
        self.test_data_list = None
        if load_dataset:
            with open(config.data_dir, 'r') as f:
                self.facet_all_data = json.load(f)

        with open(config.vocab_dir, 'r') as f:
            self.ngrams_dict = json.load(f)
        self.ngram_n = config.ngram_n

    def prepare_data(self, tracker_idx):
        facet_name = self.index2facet[tracker_idx]
        all_data = self.facet_all_data[facet_name]
        sen_list = []
        label_list = []
        for single_data in all_data:
            sen = single_data["data"]
            label = single_data["label"]
            sen_list.append(sen)
            label_list.append(label)
        data_num = len(sen_list)
        data_num = data_num // 10
        eva_sen_list = sen_list[:data_num]
        test_sen_list = sen_list[data_num:2*data_num]
        train_sen_list = sen_list[2*data_num:]
        eva_label_list = label_list[:data_num]
        test_label_list = label_list[data_num:2*data_num]
        train_label_list = label_list[2*data_num:]

        assert len(train_sen_list) == len(train_label_list)
        assert len(eva_sen_list) == len(eva_label_list)
        assert len(test_sen_list) == len(test_label_list)

        def make_batch_data(sen_list, label_list, batch_size, padding_word_idx):
            batch_data_list = []
            for index in range(len(sen_list)//batch_size):
                left_index = index * batch_size
                right_index = (index+1) * batch_size
                batch_data = BatchData(sen_list[left_index:right_index], label_list[left_index:right_index], padding_word_idx, self.use_gpu)
                batch_data_list.append(batch_data)
            return batch_data_list

        self.train_data_list = make_batch_data(train_sen_list, train_label_list, self.batch_size, self.padding_word_idx)
        self.eva_data_list = make_batch_data(eva_sen_list, eva_label_list, self.batch_size, self.padding_word_idx)
        self.test_data_list = make_batch_data(test_sen_list, test_label_list, self.batch_size, self.padding_word_idx)

    def batch_data_input(self, batchdata, tracker_idx, is_train):
        if is_train:
            self.belieftracker_list[tracker_idx].train()
        else:
            self.belieftracker_list[tracker_idx].eval()

        output = self.belieftracker_list[tracker_idx](batchdata.words_inputs, batchdata.word_seq_lengths, is_train)
        if not is_train:
            output = torch.argmax(output, -1)
        return output

    def save_all_model(self):
        for each_tracker, each_path in zip(self.belieftracker_list, self.belieftracker_model_name_list):
            torch.save(each_tracker.state_dict(), "/".join([self.belieftracker_model_path, each_path]))

    def load_all_model(self):
        for each_tracker, each_path in zip(self.belieftracker_list, self.belieftracker_model_name_list):   
            each_tracker.load_state_dict(torch.load("/".join([self.belieftracker_model_path, each_path])))

    def save_single_model(self, tracker_idx):
        torch.save(self.belieftracker_list[tracker_idx].state_dict(), \
                "/".join([self.belieftracker_model_path, self.belieftracker_model_name_list[tracker_idx]]))

    def load_single_model(self, tracker_idx):
        self.belieftracker_list[tracker_idx].load_state_dict(torch.load(\
                "/".join([self.belieftracker_model_path, self.belieftracker_model_name_list[tracker_idx]])))

    def train_model(self, tracker_idx):
        print("prepare data...")
        self.prepare_data(tracker_idx)
        time_str = datetime.datetime.now().isoformat()
        print("{} start training tracker {} ...".format(time_str, tracker_idx))
        print("lr: {:g}, batch_size: {}, weight_decay: {:g}, hidden_dim: {}"\
                .format(self.lr, self.batch_size, self.weight_decay, self.tracker_hidden_dim))
        step = 0
        best_acc = 0.
        train_log_name = "tracker_{}_".format(tracker_idx) + time_str + ".txt"
        with open("/".join([self.train_log_path, train_log_name]), "a") as f:
            f.write("{} start training tracker {} ...\n".format(time_str, tracker_idx))
            f.write("lr: {:g}, batch_size: {}, weight_decay: {:g}, hidden_dim: {}\n"\
                .format(self.lr, self.batch_size, self.weight_decay, self.tracker_hidden_dim))
            for _ in range(self.epoch_num):
                print("epoch: ", _)
                f.write("epoch: "+str(_)+'\n')
                train_data_index_list = [_ for _ in range(len(self.train_data_list))]
                random.shuffle(train_data_index_list)
                for train_data_index in train_data_index_list:
                    t_batch_data = self.train_data_list[train_data_index]
                    output = self.batch_data_input(t_batch_data, tracker_idx, True)
                    loss = self.criterion(output, t_batch_data.label_list)
                    self.optimizer_list[tracker_idx].zero_grad()
                    loss.backward()
                    self.optimizer_list[tracker_idx].step() 

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}".format(time_str, step, loss))
                    f.write("{}: step {}, loss {:g}\n".format(time_str, step, loss))

                    step += 1
                    if step % self.save_step == 0:
                        print("start evaluation")
                        pre_label_list = []
                        gt_label_list = []
                        for e_batch_data in self.eva_data_list:
                            output = self.batch_data_input(e_batch_data, tracker_idx, False)
                            pre_label_list.extend(output.tolist())
                            gt_label_list.extend(e_batch_data.label_list.tolist())
                        cur_acc = accuracy_score(gt_label_list, pre_label_list)
                        print("{}: step {}, accuracy {:g}".format(time_str, step, cur_acc))
                        f.write("----------\n")
                        f.write("Eva {}: step {}, accuracy {:g}\n".format(time_str, step, cur_acc))
                        f.write("----------\n")
                        if cur_acc > best_acc:
                            best_acc = cur_acc
                            self.save_single_model(tracker_idx)
            f.write("----------\n")
            f.write("train over\n")
            f.write("best acc: {:g}\n".format(best_acc))

    def eva_model(self, tracker_idx):
        print("prepare data...")
        self.prepare_data(tracker_idx)
        self.load_single_model(tracker_idx)
        time_str = datetime.datetime.now().isoformat()
        print("{} start test tracker {} ...".format(time_str, tracker_idx))
        pre_label_list = []
        gt_label_list = []
        for t_batch_data in self.test_data_list:
            output = self.batch_data_input(t_batch_data, tracker_idx, False)
            pre_label_list.extend(output.tolist())
            gt_label_list.extend(t_batch_data.label_list)
        test_acc = accuracy_score(gt_label_list, pre_label_list)
        test_log_name = "tracker_{}_test_".format(tracker_idx) + time_str + ".txt"
        with open("/".join([self.train_log_path, test_log_name]), "a") as f:
            f.write("best acc: {:g}\n".format(test_acc))

    def use_tracker(self, sendata, tracker_idx_list):
        batch_data = BatchData(sendata, None, self.padding_word_idx, self.use_gpu)
        output = []
        for tracker_idx in tracker_idx_list:
            self.belieftracker_list[tracker_idx].eval()
            output.append(self.belieftracker_list[tracker_idx](batch_data.words_inputs, batch_data.word_seq_lengths, False))
        output = torch.cat(output, -1)
        return output

    def use_tracker_from_nl(self, sendata_str, tracker_idx_list):
        sendata = []
        for each_sen_str in sendata_str:
            each_sen = []
            ngrams_list = generate_ngrams(each_sen_str, self.ngram_n+1)
            for ngram_ in ngrams_list:
                ngram_str = ' '.join(ngram_)
                ngrams_idx = self.ngrams_dict.get(ngram_str, self.padding_word_idx)
                each_sen.append(ngrams_idx)
            sendata.append(each_sen)
        sendata = [sendata]
        return self.use_tracker(sendata, tracker_idx_list)[0]
