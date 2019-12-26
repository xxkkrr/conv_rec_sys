import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class AgentRLConfig():
    def __init__(self):
    	self.use_gpu = False
    	self.input_dim = 189+11+189+4+9
    	self.hidden1_dim = 100
    	self.hidden2_dim = 6
    	self.dp = 0.2
    	self.DPN_model_path = root_path + "/agents/agent_rl_model/"
    	self.DPN_model_name = "TwoLayer"
    	self.all_facet_list = ["categories", "state", "city", "price", "stars"]
    	self.utt_gen_dict_path = root_path + "/data/agents/utt_gen_dict.json"
    	self.pretrain_data_path = root_path + "/data/RL_data/RL_pretrain_data.pkl"
    	self.pretrain_epoch_num = 200
    	self.pretrain_lr = 0.003
    	self.pretrain_batch_size = 64
    	self.pretrain_log_path = root_path + "/agents/"
    	self.pretrain_save_step = 200
    	self.PG_discount_rate = 0.95
    	self.PG_batch_size = 100
    	self.PG_data_path = root_path + "/data/RL_data/RL_data.pkl"
    	self.PG_epoch_num = 100
    	self.PG_lr = 0.001
    	self.PG_save_step = 10
    	self.PG_silence = True
    	self.PG_log_path = root_path + "/agents/"