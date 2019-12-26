import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class RetrieverConfig():
    def __init__(self):
        self.DB_dir = root_path + "/data/businessDB/businessDB_dict.json"
        self.value_nl_dict_dir = root_path + "/data/belief_tracker_data/restaurants_slot_value_filter.json"