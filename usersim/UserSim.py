import json
import random

class UserSim():
    def __init__(self, config):
        with open(config.utt_gen_dict_path, 'r') as f:
            self.utt_gen_dict = json.load(f)
        with open(config.restaurants_info_dict_path, 'r') as f:
            self.restaurants_info_dict = json.load(f)
        self.business_info_dict = None
        self.user_name = None
        
    def init_episode(self, user_name, business_name):
        self.business_info_dict = self.restaurants_info_dict[business_name]
        self.user_name = user_name

    def next_turn(self, request_facet):
        facet_value = self.business_info_dict[request_facet]
        pattern_dict = random.choice(self.utt_gen_dict[request_facet])
        utterance_content = pattern_dict["nl"]
        utterance_content = utterance_content.replace('$'+request_facet+'$', str(facet_value), 1)
        return utterance_content