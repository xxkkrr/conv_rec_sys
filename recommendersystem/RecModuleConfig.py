import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class RecModuleConfig():
    def __init__(self):
        self.all_facet_list = ["categories", "state", "city", "price", "stars"] # facet list
        self.facet_dim_list = [189, 11, 189, 4, 9] # value kinds of each facet
        self.max_condition_num = 3
        self.business_dict_path = root_path + "/data/FM_data/business_dict.json"
        self.user_dict_path = root_path + "/data/FM_data/user_dict.json"