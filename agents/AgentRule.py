import random
import torch
import json
from torch.distributions import Categorical

def calculate_entropy(probs_tensor):
    probs_entropy = Categorical(probs = probs_tensor).entropy()
    return probs_entropy

class AgentRule():
    def __init__(self, config):
        self.all_facet_list = config.all_facet_list
        self.facet_dim_list = config.facet_dim_list
        with open(config.utt_gen_dict_path, 'r') as f:
            self.utt_gen_dict = json.load(f)

        def build_facet_index_dict():
            facet_index_dict = {}
            start_idx = 0
            for facet, dim_num in zip(self.all_facet_list, self.facet_dim_list):
                facet_index_dict[facet] = (start_idx, start_idx + dim_num)
                start_idx += dim_num
            return facet_index_dict
        self.facet_index_dict = build_facet_index_dict()

        self.unknown_facet = []

    def init_episode(self):
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

        if len(self.unknown_facet) == 0:
            return "recommend", "recommend"

        facet_entropy_list = []
        for facet in self.all_facet_list:
            facet_entropy = torch.tensor(0.)
            if facet in self.unknown_facet:
                left_index = self.facet_index_dict[facet][0]
                right_index = self.facet_index_dict[facet][1]
                facet_distribution = dialogue_state[left_index:right_index]
                facet_entropy = calculate_entropy(facet_distribution)
            facet_entropy_list.append(facet_entropy)
        facet_entropy_tensor = torch.stack(facet_entropy_list)
        facet_index = int(facet_entropy_tensor.argmax())
        current_facet = self.all_facet_list[facet_index]
        self.unknown_facet.remove(current_facet)
        return current_facet, self.generate_nl(current_facet)

    def current_unknown_facet(self):
        return self.unknown_facet