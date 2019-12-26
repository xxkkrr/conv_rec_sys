import json

class Retriever():
    def __init__(self, config):
        with open(config.DB_dir, 'r') as f:
            self.businessDB_dict = json.load(f)
        with open(config.value_nl_dict_dir, 'r') as f:
            self.value_nl_dict = json.load(f)
        self.value2nl_dict = {}
        for facet in self.value_nl_dict.keys():
            value2nl = {}
            for key,value in self.value_nl_dict[facet].items():
                value2nl[value] = key
            self.value2nl_dict[facet] = value2nl

    def filter_bussiness(self, condition_dict_list):
        condition_dict_nl_list = []
        for condition_dict in condition_dict_list:
            condition_dict_nl = {}
            for facet in condition_dict.keys():
                condition_dict_nl[facet] = self.value2nl_dict[facet][condition_dict[facet]]
            condition_dict_nl_list.append(condition_dict_nl)
        # print("condition_dict_list:", condition_dict_nl_list)
        def check_satisfy(bussiness_dict):
            satisfy_num = len(condition_dict_list[0])
            for condition_dict in condition_dict_list:
                equal_num = 0
                for facet in condition_dict.keys():
                    if condition_dict[facet] == bussiness_dict[facet]:
                        equal_num += 1
                    else:
                        break
                if equal_num == satisfy_num:
                    return True
            return False

        bussiness_list = []
        for bussiness_id in self.businessDB_dict.keys():
            if check_satisfy(self.businessDB_dict[bussiness_id]):
                bussiness_list.append(int(bussiness_id))
        return bussiness_list