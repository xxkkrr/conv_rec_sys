import torch
import torch.nn as nn
import json
from recommendersystem.FMConfig import FMConfig
from recommendersystem.FMModule import FMModule
from recommendersystem.Retriever import Retriever
from recommendersystem.RetrieverConfig import RetrieverConfig

class RecModule():
    def __init__(self, config):
        self.all_facet_list = config.all_facet_list
        self.facet_dim_list = config.facet_dim_list
        def build_facet_index_dict():
            facet_index_dict = {}
            start_idx = 0
            for facet, dim_num in zip(self.all_facet_list, self.facet_dim_list):
                facet_index_dict[facet] = (start_idx, start_idx + dim_num)
                start_idx += dim_num
            return facet_index_dict
        self.facet_index_dict = build_facet_index_dict()
        self.max_condition_num = config.max_condition_num
        with open(config.business_dict_path, 'r') as f:
            self.business_dict = json.load(f)
        self.reverse_business_dict = {}
        for key,value in self.business_dict.items():
            self.reverse_business_dict[value] = key
        with open(config.user_dict_path, 'r') as f:
            self.user_dict = json.load(f)       
        self.FM = FMModule(FMConfig(), False)
        self.FM.load_model()
        self.retriever = Retriever(RetrieverConfig())

    def get_condition_dict_list(self, dialogue_state, unknown_facet):
        facet_value_probs_dict = {}
        known_facet_list = []
        for facet in self.all_facet_list:
            if facet in unknown_facet:
                continue
            known_facet_list.append(facet)
            value_probs_dict = {}
            left_index = self.facet_index_dict[facet][0]
            right_index = self.facet_index_dict[facet][1]
            facet_distribution = dialogue_state[left_index:right_index]
            for value_index, prob in enumerate(facet_distribution):
                value_probs_dict[str(value_index)] = prob
            facet_value_probs_dict[facet] = list(value_probs_dict.items())

        def merge_two(list_a, list_b, limit_num):
            list_b = sorted(list_b, key=lambda x:x[1], reverse=True)
            list_b = list_b[:limit_num]
            if len(list_a) == 0:
                return list_b
            list_c = []
            for tuple_a in list_a:
                for tuple_b in list_b:
                    tuple_c = []
                    tuple_c.append('&'.join([tuple_a[0],tuple_b[0]]))
                    tuple_c.append(tuple_a[1]*tuple_b[1])
                    list_c.append(tuple(tuple_c))
            list_c = sorted(list_c, key=lambda x:x[1], reverse=True)
            return list_c[:limit_num]

        condition_list = []
        for facet in known_facet_list:
            condition_list = merge_two(condition_list, facet_value_probs_dict[facet], self.max_condition_num)
        condition_dict_list = []
        for condition in condition_list:
            value_list = condition[0].split('&')
            condition_dict = {}
            for facet, value in zip(known_facet_list, value_list):
                condition_dict[facet] = int(value)
            condition_dict_list.append(condition_dict)
        return condition_dict_list

    def recommend_bussiness(self, user_str, dialogue_state, unknown_facet):
        user_id = self.user_dict[user_str]
        condition_dict_list = self.get_condition_dict_list(dialogue_state, unknown_facet)
        business_list = self.retriever.filter_bussiness(condition_dict_list)
        if len(business_list) == 0:
             return business_list
        ranked_business_list = self.FM.use_FM(user_id, business_list, dialogue_state)
        return list(map(lambda x: self.reverse_business_dict[x], ranked_business_list))