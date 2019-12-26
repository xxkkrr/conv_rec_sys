import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import datetime
import random
random.seed(9001)
import numpy as np
from recommendersystem.FM import TorchFM
from belieftracker.BeliefTrackerModule import BeliefTrackerModule
from belieftracker.BeliefTrackerConfig import BeliefTrackerConfig

class BatchData():
    def __init__(self, data_list, user_num, business_num, use_gpu, output_check=False):
        self.sen_list = [_["data"] for _ in data_list]
        
        user_list = torch.tensor([_["user_id"] for _ in data_list])
        business_list = torch.tensor([_["business_id"] for _ in data_list])
        user_list_onehot = nn.functional.one_hot(user_list, user_num)
        business_list_onehot = nn.functional.one_hot(business_list, business_num)
        self.user_business_feature = torch.cat([user_list_onehot, business_list_onehot], -1).float()

        self.score_list = torch.tensor([float(_["score"]) for _ in data_list])

        if use_gpu:
            self.user_business_feature = self.user_business_feature.cuda()
            self.score_list = self.score_list.cuda()

        if output_check:
            self.output()

    def output(self):
        print("--------------------------")
        print("sen_list:", self.sen_list)
        print("user_business_feature:", self.user_business_feature)
        print("score_list:", self.score_list)

class FMModule():
    def __init__(self, config, load_dataset):
        self.FM_model_path = config.FM_model_path
        self.data_dir = config.data_dir
        self.user_num = config.user_num
        self.business_num = config.business_num
        self.BeliefTrackerModule = BeliefTrackerModule(BeliefTrackerConfig(), False)
        self.BeliefTrackerModule.load_all_model()
        self.tracker_idx_list = config.tracker_idx_list
        self.FM = TorchFM(config)
        self.batch_size = config.batch_size
        self.use_gpu = config.use_gpu
        self.epoch_num = config.epoch_num
        self.save_step = config.save_step
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.optimizer = optim.Adam(self.FM.parameters(), lr=self.lr,)
        self.train_log_path = config.train_log_path
        self.criterion = nn.MSELoss(reduce=True, size_average=True)

        self.FM_all_data = None
        self.train_data_list = None
        self.eva_data_list = None
        self.test_data_list = None
        if load_dataset:
            with open(config.data_dir, 'r') as f:
                self.FM_all_data = json.load(f)

    def save_model(self):
        torch.save(self.FM.state_dict(), "/".join([self.FM_model_path, "FM_model"]))

    def load_model(self):
        self.FM.load_state_dict(torch.load("/".join([self.FM_model_path, "FM_model"])))

    def prepare_data(self):
        random.shuffle(self.FM_all_data)
        data_num = len(self.FM_all_data) // 10
        eva_list = self.FM_all_data[:data_num]
        test_list = self.FM_all_data[data_num:2*data_num]
        train_list = self.FM_all_data[2*data_num:]

        # def make_batch_data(data_list, batch_size):
        #     batch_data_list = []
        #     for index in range(len(data_list)//batch_size):
        #         left_index = index * batch_size
        #         right_index = (index+1) * batch_size
        #         batch_data = BatchData(data_list[left_index:right_index], \
        #                                 self.user_num, self.business_num, self.use_gpu)
        #         batch_data_list.append(batch_data)
        #     return batch_data_list

        # self.train_data_list = make_batch_data(train_list, self.batch_size)
        # self.eva_data_list = make_batch_data(eva_list, self.batch_size)
        # self.test_data_list = make_batch_data(test_list, self.batch_size)      
        self.train_data_list = train_list
        self.eva_data_list = eva_list
        self.test_data_list = test_list

    def batch_data_input(self, batchdata, is_train):
        if is_train:
            self.FM.train()
        else:
            self.FM.eval()

        tracker_output = self.BeliefTrackerModule.use_tracker(batchdata.sen_list, self.tracker_idx_list)
        input_feature = torch.cat([batchdata.user_business_feature, tracker_output], -1)
        output = self.FM(input_feature)
        return output

    def train_model(self):
        print("prepare data...")
        self.prepare_data()
        time_str = datetime.datetime.now().isoformat()
        print("{} start training FM ...".format(time_str))
        print("lr: {:g}, batch_size: {}, weight_decay: {:g}"\
                .format(self.lr, self.batch_size, self.weight_decay))
        step = 0
        min_loss = 1e10
        train_log_name = "FM_" + time_str + ".txt"
        with open("/".join([self.train_log_path, train_log_name]), "a") as f:
            f.write("{} start training FM ...\n".format(time_str))
            f.write("lr: {:g}, batch_size: {}, weight_decay: {:g}\n"\
                .format(self.lr, self.batch_size, self.weight_decay))
            for _ in range(self.epoch_num):
                print("epoch: ", _)
                f.write("epoch: "+str(_)+'\n')
                # train_data_index_list = [_ for _ in range(len(self.train_data_list))]
                # random.shuffle(train_data_index_list)
                random.shuffle(self.train_data_list)
                # for train_data_index in train_data_index_list:
                for index in range(len(self.train_data_list)//self.batch_size):
                    left_index = index * self.batch_size
                    right_index = (index+1) * self.batch_size
                    t_batch_data = BatchData(self.train_data_list[left_index:right_index], \
                                            self.user_num, self.business_num, self.use_gpu)

                    output = self.batch_data_input(t_batch_data, True)
                    loss = self.criterion(output, t_batch_data.score_list)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step() 

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}".format(time_str, step, loss))
                    f.write("{}: step {}, loss {:g}\n".format(time_str, step, loss))

                    step += 1
                    if step % self.save_step == 0:
                        print("start evaluation")
                        # for e_batch_data in self.eva_data_list:
                        cur_loss = 0.
                        for index_ in range(len(self.eva_data_list)//self.batch_size):
                            left_index = index_ * self.batch_size
                            right_index = (index_+1) * self.batch_size
                            e_batch_data = BatchData(self.eva_data_list[left_index:right_index], \
                                                    self.user_num, self.business_num, self.use_gpu)
                            output = self.batch_data_input(e_batch_data, False)
                            pre_score_list = output.view(-1).tolist()
                            gt_score_list = e_batch_data.score_list.tolist()
                            for pre, gt in zip(pre_score_list, gt_score_list):
                                cur_loss += (pre - gt) ** 2
                        print("{}: step {}, score loss {:g}".format(time_str, step, cur_loss))
                        f.write("----------\n")
                        f.write("Eva {}: step {}, score loss {:g}\n".format(time_str, step, cur_loss))
                        f.write("----------\n")
                        if cur_loss < min_loss:
                            min_loss = cur_loss
                            self.save_model()
            f.write("----------\n")
            f.write("train over\n")
            f.write("min loss: {:g}\n".format(min_loss))

    def eva_model(self):
        print("prepare data...")
        self.prepare_data()
        self.load_model()
        time_str = datetime.datetime.now().isoformat()
        print("{} start test FM ...".format(time_str))
        test_loss = 0.
        # for t_batch_data in self.test_data_list:
        for index in range(len(self.test_data_list)//self.batch_size):
            left_index = index * self.batch_size
            right_index = (index+1) * self.batch_size
            t_batch_data = BatchData(self.test_data_list[left_index:right_index], \
                                    self.user_num, self.business_num, self.use_gpu)
            output = self.batch_data_input(t_batch_data, False)
            pre_score_list = output.view(-1).tolist()
            gt_score_list = t_batch_data.score_list.tolist()
            for pre, gt in zip(pre_score_list, gt_score_list):
                test_loss += (pre - gt) ** 2        
        test_log_name = "FM_test_" + time_str + ".txt"
        with open("/".join([self.train_log_path, test_log_name]), "a") as f:
            f.write("test_loss: {:g}\n".format(test_loss))

    def use_FM(self, user_id, business_list, current_state):
        user_list = torch.tensor([user_id] * len(business_list))
        business_list = torch.tensor(business_list)
        business_list_ = torch.tensor(business_list)
        user_list = nn.functional.one_hot(user_list, self.user_num).float()
        business_list = nn.functional.one_hot(business_list, self.business_num).float()
        current_state_list = torch.stack([current_state for _ in range(len(business_list))], 0)
        input_list = torch.cat([user_list, business_list, current_state_list], -1)
        score_list = self.FM(input_list)
        _, sorted_indices = score_list.sort(descending=True)
        return business_list_[sorted_indices].tolist()