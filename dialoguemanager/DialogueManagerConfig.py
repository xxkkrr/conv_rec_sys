class DialogueManagerConfig():
    def __init__(self):
        self.K = 30
        self.C = 40.
        self.rc = -1.
        self.rq = -10.
        self.turn_limit = 7
        self.tracker_idx_list = [0, 1, 2, 3, 4]
        self.rec_action_facet = "recommend"