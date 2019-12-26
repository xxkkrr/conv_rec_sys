import json
import random

class UserReal():
    def __init__(self, config):
        with open(config.restaurants_info_dict_path, 'r') as f:
            self.restaurants_info_dict = json.load(f)
        self.business_info_dict = None
        self.user_name = None
        
    def init_episode(self, user_name, business_name):
        self.business_info_dict = self.restaurants_info_dict[business_name]
        self.user_name = user_name
        print("Now you are user {}".format(self.user_name))
        print("Target restaurant info: {}".format(str(self.business_info_dict)))

    def next_turn(self, request_facet):
        utterance_content = input("your turn:")
        return utterance_content