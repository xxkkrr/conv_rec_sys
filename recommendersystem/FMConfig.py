import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class FMConfig():
    def __init__(self):
        self.use_gpu = False # use gpu or not, choose from [True,False]
        self.FM_model_path = root_path + "/recommendersystem/FM_model_save" # root path of FM model
        self.FM_n = 68864 + 19951 + 189 + 11 + 189 + 4 + 9
        self.FM_k = 2
        self.data_dir = root_path + "/data/FM_data/FM_data_list.json" # path to data
        self.user_num = 68864 # user number
        self.business_num = 19951 # business number
        self.tracker_idx_list = [0, 1, 2, 3, 4] # use tracker idx list
        self.batch_size = 32 # batch size used in training
        self.epoch_num = 5 # epoch num used in training
        self.save_step = 1000 # save model after each save_step when training
        self.lr = 1e-3 # learning rate used in training
        self.weight_decay = 0. # weight decay used in training
        self.train_log_path = "./" # train log sava path