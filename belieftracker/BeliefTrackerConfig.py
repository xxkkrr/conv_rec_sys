import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class BeliefTrackerConfig():
    def __init__(self):
        self.use_gpu = False # use gpu or not, choose from [True,False]
        self.word_embedding_dim = 300 # word_embedding_dim
        self.ngram_embedding = True # use ngram or not, choose from [True,False]
        self.dp = 0.2 # drop out rate
        self.word_alphabet_size = 2218 # word num
        self.pretrain_word_embedding = None # pretrain word embedding path
        self.bilstm = True # use bi-direction or not, choose from [True,False]
        self.layer_num = 1 # the layer num of lstm
        self.hidden_dim = 300 # the hidden dim of lstm
        self.word_feature_extractor = "LSTM" # choose from ["GRU","LSTM"]
        self.belieftracker_model_path = root_path + "/belieftracker/belieftracker_model_save" # root path of belieftracker model
        self.belieftracker_model_name_list = ["categories", "state", "city", "price", "stars"] # name of differrent belieftracker model
        self.data_dir = root_path + "/data/belief_tracker_data/aspect_all_data.json" # path to data
        self.batch_size = 64 # batch size used in training
        self.value_num_list = [189, 11, 189, 4, 9] # num of value kinds of differrent aspect
        self.index2facet = {0:"categories", 1:"state", 2:"city", 3:"price", 4:"stars"} # map belieftracker index to aspect
        self.epoch_num = 1 # epoch num used in training
        self.save_step = 1000 # save model after each save_step when training
        self.lr = 5e-2 # learning rate used in training
        self.weight_decay = 0. # weight decay used in training
        self.train_log_path = "./" # train log sava path
        self.vocab_dir = root_path + "/data/belief_tracker_data/vocab_dict.json" # path to ngram dict
        self.ngram_n = 2