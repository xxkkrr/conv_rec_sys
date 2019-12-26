import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import random
import pickle
import datetime
from agents.DeepPolicyNetwork import TwoLayersModel
from agents.RLEnv import RLEnv
from agents.RLEnvConfig import RLEnvConfig

class PretrainBatchData():
    def __init__(self, state_list, label_list, use_gpu, output_check=False):
        batch_size = len(state_list)
        self.state_list = torch.tensor(state_list)
        self.label_list = torch.tensor(label_list)
        if use_gpu:
            self.state_list = self.state_list.cuda()
            self.label_list = self.label_list.cuda()
        if output_check:
            self.output()

    def output(self):
        print("--------------------------")
        print("state_list:", self.state_list)
        print("label_list:", self.label_list)

class AgentRL():
    def __init__(self, config):
        self.use_gpu = config.use_gpu
        self.DPN = TwoLayersModel(config)
        self.DPN_model_path = config.DPN_model_path
        self.DPN_model_name = config.DPN_model_name
        self.all_facet_list = config.all_facet_list
        with open(config.utt_gen_dict_path, 'r') as f:
            self.utt_gen_dict = json.load(f)
        self.unknown_facet = []

        self.pretrain_data_path = config.pretrain_data_path
        self.pretrain_epoch_num = config.pretrain_epoch_num
        self.pretrain_lr = config.pretrain_lr
        self.pretrain_optimizer = optim.Adam(self.DPN.parameters(), lr=self.pretrain_lr,)
        self.pretrain_criterion = nn.CrossEntropyLoss()
        self.pretrain_batch_size = config.pretrain_batch_size
        self.pretrain_log_path = config.pretrain_log_path
        self.pretrain_save_step = config.pretrain_save_step
        self.pretrain_train_data_list = None
        self.pretrain_eva_data_list = None
        self.pretrain_test_data_list = None

        self.env = None
        self.PG_lr = config.PG_lr
        self.PG_discount_rate = config.PG_discount_rate
        self.PG_batch_size = config.PG_batch_size
        self.PG_data_path = config.PG_data_path
        self.PG_epoch_num = config.PG_epoch_num
        self.PG_optimizer = optim.RMSprop(self.DPN.parameters(), lr=self.PG_lr,)
        self.PG_log_path = config.PG_log_path
        self.PG_save_step = config.PG_save_step
        self.PG_silence = config.PG_silence
        self.PG_train_data_list = None
        self.PG_eva_data_list = None
        self.PG_test_data_list = None

    def init_episode(self):
        self.DPN.eval()
        self.unknown_facet = set(self.all_facet_list)

    def generate_nl(self, facet):
        pattern_dict = random.choice(self.utt_gen_dict[facet])
        utterance_content = pattern_dict['nl']
        return utterance_content

    def next_turn(self, dialogue_state):
        if type(dialogue_state) == type(None):
            current_facet = self.all_facet_list[0]
            self.unknown_facet.remove(current_facet)
            return current_facet, self.generate_nl(current_facet)

        facet_distribution = self.DPN(dialogue_state, True)
        facet_index = int(facet_distribution.argmax())
        if facet_index == len(self.all_facet_list):
            return "recommend", "recommend"
        current_facet = self.all_facet_list[facet_index]
        if current_facet in self.unknown_facet:
            self.unknown_facet.remove(current_facet)
        return current_facet, self.generate_nl(current_facet)

    def current_unknown_facet(self):
        return self.unknown_facet

    def save_model(self, is_preatrain):
        if is_preatrain:
            name_suffix = "_PRE"
        else:
            name_suffix = "_PG"
        torch.save(self.DPN.state_dict(), "/".join([self.DPN_model_path, self.DPN_model_name + name_suffix]))

    def load_model(self, is_preatrain):
        if is_preatrain:
            name_suffix = "_PRE"
        else:
            name_suffix = "_PG"
        self.DPN.load_state_dict(torch.load("/".join([self.DPN_model_path, self.DPN_model_name + name_suffix])))

    def load_pretrain_data(self):
        with open(self.pretrain_data_path, 'rb') as f:
            dialogue_state_list, label_list = pickle.load(f)
        assert len(dialogue_state_list) == len(label_list)

        data_num = len(dialogue_state_list)
        data_num = data_num // 10
        eva_state_list = dialogue_state_list[:data_num]
        test_state_list = dialogue_state_list[data_num:2*data_num]
        train_state_list = dialogue_state_list[2*data_num:]
        eva_label_list = label_list[:data_num]
        test_label_list = label_list[data_num:2*data_num]
        train_label_list = label_list[2*data_num:]     

        def make_batch_data(state_list, label_list, batch_size):
            batch_data_list = []
            for index in range(len(state_list)//batch_size):
                left_index = index * batch_size
                right_index = (index+1) * batch_size
                batch_data = PretrainBatchData(state_list[left_index:right_index], label_list[left_index:right_index], self.use_gpu)
                batch_data_list.append(batch_data)
            return batch_data_list

        self.pretrain_train_data_list = make_batch_data(train_state_list, train_label_list, self.pretrain_batch_size)
        self.pretrain_eva_data_list = make_batch_data(eva_state_list, eva_label_list, self.pretrain_batch_size)
        self.pretrain_test_data_list = make_batch_data(test_state_list, test_label_list, self.pretrain_batch_size)

    def pretrain_batch_data_input(self, batchdata, is_train):
        if is_train:
            self.DPN.train()
        else:
            self.DPN.eval()

        output = self.DPN(batchdata.state_list, is_train)
        if not is_train:
            output = torch.argmax(output, -1)
        return output

    def pretrain_train(self):
        print("prepare pretrain data...")
        self.load_pretrain_data()
        time_str = datetime.datetime.now().isoformat()
        print("{} start pretraining ...".format(time_str))
        print("lr: {:g}, batch_size: {}".format(self.pretrain_lr, self.pretrain_batch_size))
        step = 0
        best_acc = 0.
        pretrain_log_name = "DPN_pretrain_" + time_str + ".txt"
        with open("/".join([self.pretrain_log_path, pretrain_log_name]), "a") as f:
            f.write("{} start pretraining ...\n".format(time_str))
            f.write("lr: {:g}, batch_size: {}\n".format(self.pretrain_lr, self.pretrain_batch_size))
            for _ in range(self.pretrain_epoch_num):
                print("epoch: ", _)
                f.write("epoch: "+str(_)+'\n')
                pretrain_data_index_list = [_ for _ in range(len(self.pretrain_train_data_list))]
                random.shuffle(pretrain_data_index_list)
                for pretrain_data_index in pretrain_data_index_list:
                    t_batch_data = self.pretrain_train_data_list[pretrain_data_index]
                    output = self.pretrain_batch_data_input(t_batch_data, True)
                    loss = self.pretrain_criterion(output, t_batch_data.label_list)
                    self.pretrain_optimizer.zero_grad()
                    loss.backward()
                    self.pretrain_optimizer.step()

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}".format(time_str, step, loss))
                    f.write("{}: step {}, loss {:g}\n".format(time_str, step, loss))

                    step += 1
                    if step % self.pretrain_save_step == 0:
                        print("start evaluation")
                        pre_label_list = []
                        gt_label_list = []
                        for e_batch_data in self.pretrain_eva_data_list:
                            output = self.pretrain_batch_data_input(e_batch_data, False)
                            pre_label_list.extend(output.tolist())
                            gt_label_list.extend(e_batch_data.label_list.tolist())
                        cur_acc = accuracy_score(gt_label_list, pre_label_list)
                        print("{}: step {}, accuracy {:g}".format(time_str, step, cur_acc))
                        f.write("----------\n")
                        f.write("Eva {}: step {}, accuracy {:g}\n".format(time_str, step, cur_acc))
                        f.write("----------\n")
                        if cur_acc > best_acc:
                            best_acc = cur_acc
                            self.save_model(True)
            f.write("----------\n")
            f.write("train over\n")
            f.write("best acc: {:g}\n".format(best_acc))

    def pretrain_eva(self):
        print("prepare pretrain data...")
        self.load_pretrain_data()
        self.load_model(True)
        time_str = datetime.datetime.now().isoformat()
        print("{} start pretraining Test ...".format(time_str))
        pre_label_list = []
        gt_label_list = []
        for t_batch_data in self.pretrain_test_data_list:
            output = self.pretrain_batch_data_input(t_batch_data, False)
            pre_label_list.extend(output.tolist())
            gt_label_list.extend(t_batch_data.label_list)
        test_acc = accuracy_score(gt_label_list, pre_label_list)
        pretrain_test_log_name = "pretrain_test_" + time_str + ".txt"
        with open("/".join([self.pretrain_log_path, pretrain_test_log_name]), "a") as f:
            f.write("best acc: {:g}\n".format(test_acc))

    def load_PG_data(self):
        with open(self.PG_data_path, 'rb') as f:
            self.PG_train_data_list, self.PG_eva_data_list, \
                self.PG_test_data_list = pickle.load(f)

    def PG_train_one_episode(self, t_data):
        self.DPN.train()
        self.unknown_facet = set(self.all_facet_list)
        state_pool = []
        action_pool = []
        reward_pool = [] 

        state = self.env.initialize_episode(t_data[0], t_data[1], self.PG_silence)
        IsOver = False
        while not IsOver:
            if type(state) == type(None):
                action = 0
                current_facet = self.all_facet_list[action]
                self.unknown_facet.remove(current_facet)
                IsOver, state, reward = self.env.step(current_facet, self.unknown_facet)
            else:
                facet_distribution = self.DPN(state, False)
                c = Categorical(probs = facet_distribution)
                action = c.sample()
                if int(action) == len(self.all_facet_list):
                    current_facet = "recommend"
                else:
                    current_facet = self.all_facet_list[action]
                if current_facet in self.unknown_facet:
                    self.unknown_facet.remove(current_facet)
                IsOver, next_state, reward = self.env.step(current_facet, self.unknown_facet)                
                state_pool.append(state)
                action_pool.append(c.log_prob(action))
                reward_pool.append(reward)
                if not IsOver:
                    state = next_state

        return action_pool, reward_pool

    def PG_eva_one_episode(self, t_data):
        self.DPN.eval()
        self.unknown_facet = set(self.all_facet_list)
        total_reward = 0.
        turn_count = 0
        success = 0

        state = self.env.initialize_episode(t_data[0], t_data[1], self.PG_silence)
        IsOver = False
        while not IsOver:
            turn_count += 1
            if type(state) == type(None):
                action = 0
                current_facet = self.all_facet_list[action]
                self.unknown_facet.remove(current_facet)
                IsOver, state, reward = self.env.step(current_facet, self.unknown_facet)
                total_reward += reward
            else:
                facet_distribution = self.DPN(state, True)
                action = int(facet_distribution.argmax())
                if int(action) == len(self.all_facet_list):
                    current_facet = "recommend"
                else:
                    current_facet = self.all_facet_list[action]
                if current_facet in self.unknown_facet:
                    self.unknown_facet.remove(current_facet)
                IsOver, next_state, reward = self.env.step(current_facet, self.unknown_facet) 
                total_reward += reward
                if not IsOver:
                    state = next_state
                else:
                    if reward > 0.:
                        success = 1
                    else:
                        success = 0

        return total_reward, turn_count, success

    def PG_train(self, load_model_type):
        if load_model_type == "pretrain":
            print("load pretrain model ...")
            self.load_model(True)
        elif load_model_type == "PG":
            print("load PG model ...")
            self.load_model(False)
        else:
            print("no pretrian model...")

        self.env = RLEnv(RLEnvConfig())
        print("prepare PG data...")
        self.load_PG_data()
        time_str = datetime.datetime.now().isoformat()
        print("{} start PG ...".format(time_str))
        print("lr: {:g}, batch_size: {}".format(self.PG_lr, self.PG_batch_size))

        step = 0
        best_average_reward = 0.
        best_average_turn = 0.
        best_success_rate = 0.
        action_pool = []
        reward_pool = []
        PG_log_name = "DPN_PG_" + time_str + ".txt"
        with open("/".join([self.PG_log_path, PG_log_name]), "a") as f:
            f.write("{} start pretraining ...\n".format(time_str))
            f.write("lr: {:g}, batch_size: {}\n".format(self.PG_lr, self.PG_batch_size))
            for _ in range(self.PG_epoch_num):
                print("epoch: ", _)
                f.write("epoch: "+str(_)+'\n')
                PG_data_index_list = [_ for _ in range(len(self.PG_train_data_list))]
                random.shuffle(PG_data_index_list)
                for PG_data_index in PG_data_index_list:
                    t_data = self.PG_train_data_list[PG_data_index]
                    step += 1
                    action_pool_, reward_pool_ = self.PG_train_one_episode(t_data)
                    action_pool.append(action_pool_)
                    reward_pool.append(reward_pool_)

                    if step % self.PG_batch_size == 0:
                        for each_reward_pool in reward_pool:
                            total_reward = 0.
                            for index in reversed(range(len(each_reward_pool))):
                                total_reward = total_reward * self.PG_discount_rate + each_reward_pool[index]
                                each_reward_pool[index] = total_reward

                        # 2d to 1d
                        reward_pool = sum(reward_pool, [])
                        action_pool = sum(action_pool, [])
                        reward_pool_tensor = torch.tensor(reward_pool)
                        action_pool_tensor = torch.stack(action_pool, 0)
                        reward_pool = []
                        action_pool = []
                        if self.use_gpu:
                            reward_pool_tensor = reward_pool_tensor.cuda()
                            action_pool_tensor = action_pool_tensor.cuda()

                        loss = torch.sum(torch.mul(action_pool_tensor, reward_pool_tensor).mul(-1), -1) / self.PG_batch_size
                        self.PG_optimizer.zero_grad()
                        loss.backward()
                        self.PG_optimizer.step()

                        time_str = datetime.datetime.now().isoformat()
                        print("{}: step {}, loss {:g}".format(time_str, step, loss))
                        f.write("{}: step {}, loss {:g}\n".format(time_str, step, loss))

                        current_update_times = step/self.PG_batch_size
                        if current_update_times%self.PG_save_step == 0:
                            print("start evaluation")
                            sum_reward = 0.
                            sum_turn = 0
                            sum_success = 0
                            episode_num = 0
                            for e_data in self.PG_eva_data_list:
                                reward, turn, success = self.PG_eva_one_episode(e_data)
                                episode_num += 1
                                sum_reward += reward
                                sum_turn += turn
                                sum_success += success

                            average_reward = float(sum_reward)/episode_num
                            average_turn = float(sum_turn)/episode_num
                            success_rate = float(sum_success)/episode_num
                            time_str = datetime.datetime.now().isoformat()
                            print("{}: step {}, average_reward {:g}, average_turn {:g}, success_rate {:g}"\
                                    .format(time_str, step, average_reward, average_turn, success_rate))
                            f.write("----------\n")
                            f.write("Eva {}: step {}, average_reward {:g}, average_turn {:g}, success_rate {:g}\n"\
                                    .format(time_str, step, average_reward, average_turn, success_rate))
                            f.write("----------\n")
                            if average_reward > best_average_reward:
                                best_average_reward = average_reward
                                best_average_turn = average_turn
                                best_success_rate = success_rate
                                self.save_model(False)
            f.write("----------\n")
            f.write("train over\n")
            f.write("best_average_reward: {:g}, best_average_turn {:g}, best_success_rate {:g}\n".\
                format(best_average_reward, best_average_turn, best_success_rate))

    def PG_eva(self):
        self.env = RLEnv(RLEnvConfig())
        print("prepare PG data...")
        self.load_PG_data()
        self.load_model(False)
        time_str = datetime.datetime.now().isoformat()
        print("{} start PG Test ...".format(time_str))

        sum_reward = 0.
        sum_turn = 0
        sum_success = 0
        episode_num = 0
        for t_data in self.PG_test_data_list:
            reward, turn, success = self.PG_eva_one_episode(t_data)
            episode_num += 1
            sum_reward += reward
            sum_turn += turn
            sum_success += success

        average_reward = float(sum_reward)/episode_num
        average_turn = float(sum_turn)/episode_num
        success_rate = float(sum_success)/episode_num
        PG_test_log_name = "PG_test_" + time_str + ".txt"
        with open("/".join([self.PG_log_path, PG_test_log_name]), "a") as f:
            f.write("average_reward: {:g}, average_turn {:g}, success_rate {:g}\n".\
                format(average_reward, average_turn, success_rate))
