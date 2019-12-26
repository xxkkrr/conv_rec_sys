

class DialogueManager:
    def __init__(self, config, rec, agent, user, bftracker):
        self.K = config.K
        self.C = config.C
        self.rc = config.rc
        self.rq = config.rq
        self.turn_limit = config.turn_limit
        self.tracker_idx_list = config.tracker_idx_list
        self.rec_action_facet = config.rec_action_facet
        self.rec = rec
        self.agent = agent
        self.user = user
        self.bftracker = bftracker
        self.turn_count = None
        self.user_name = None
        self.business_name = None
        self.user_utt_list = None
        self.dialogue_state = None

    def initialize_episode(self, user_name, business_name):
        self.user_name = user_name
        self.business_name = business_name
        self.turn_count = 0
        self.agent.init_episode()
        self.user.init_episode(user_name, business_name)
        self.user_utt_list = []
        self.dialogue_state = None

    def agent_turn(self):
        request_facet, agent_nl = self.agent.next_turn(self.dialogue_state)
        print("Turn %d agent: %s" % (self.turn_count, agent_nl))
        return request_facet, agent_nl

    def user_turn(self, request_facet):
        user_nl = self.user.next_turn(request_facet)
        self.user_utt_list.append(user_nl)
        print("Turn %d user: %s" % (self.turn_count, user_nl))
        return user_nl

    def get_dialogue_state(self):
        self.dialogue_state = self.bftracker.use_tracker_from_nl(self.user_utt_list, self.tracker_idx_list)

    def recommend(self):
        business_list = self.rec.recommend_bussiness(self.user_name, self.dialogue_state, self.agent.current_unknown_facet())
        business_list = business_list[:self.K]
        for rank_index, business_name in enumerate(business_list):
            rank_id = rank_index + 1
            if business_name == self.business_name:
                rec_reward = self.C * (self.K - rank_id + 1) / self.K
                return rec_reward, rank_id, business_list
        rec_reward = self.rq
        return rec_reward, -1, business_list

    def next_turn(self):
        self.turn_count += 1 
        request_facet, agent_nl = self.agent_turn()
        if request_facet == self.rec_action_facet:
            rec_reward, rec_rank, rec_list = self.recommend()
            return True, rec_reward, rec_rank, rec_list
        if self.turn_count == self.turn_limit:
            return True, self.rq, -1, None
        user_nl = self.user_turn(request_facet)
        self.get_dialogue_state()
        return False, self.rc, None, None




