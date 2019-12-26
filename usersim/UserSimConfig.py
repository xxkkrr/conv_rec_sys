import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class UserSimConfig():
    def __init__(self):
    	self.utt_gen_dict_path = root_path + "/data/user_sim/utt_gen_dict.json"
    	self.restaurants_info_dict_path = root_path + "/data/businessDB/restaurants_info_filter.json"