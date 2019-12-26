from usersim.UserSim import UserSim
from usersim.UserSimConfig import UserSimConfig
from recommendersystem.RecModule import RecModule
from recommendersystem.RecModuleConfig import RecModuleConfig
from belieftracker.BeliefTrackerModule import BeliefTrackerModule
from belieftracker.BeliefTrackerConfig import BeliefTrackerConfig

class RLEnv:
    def __init__(self, config):
        self.K = config.K
        self.C = config.C
        self.rc = config.rc
        self.rq = config.rq
        self.turn_limit = config.turn_limit
        self.tracker_idx_list = config.tracker_idx_list
        self.rec_action_facet = config.rec_action_facet
        self.rec = RecModule(RecModuleConfig())
        self.user = UserSim(UserSimConfig())
        self.bftracker = BeliefTrackerModule(BeliefTrackerConfig(), True)
        self.bftracker.load_all_model()
        self.turn_count = None
        self.user_name = None
        self.business_name = None
        self.user_utt_list = None
        self.dialogue_state = None
        self.silence = True

    def initialize_episode(self, user_name, business_name, silence):
        if not silence:
            print("---------------------------")
            print("Simulated Conversation Start")
            print("UserName: {}, BussinessName: {}".format(user_name, business_name))
        self.user_name = user_name
        self.business_name = business_name
        self.turn_count = 0
        self.user.init_episode(user_name, business_name)
        self.user_utt_list = []
        self.dialogue_state = None
        self.silence = silence
        return self.dialogue_state

    def step(self, request_facet, unknown_facet):
        self.turn_count += 1
        if not self.silence:
            print("Turn %d agent: request %s" % (self.turn_count, request_facet))
        if request_facet == self.rec_action_facet:
            rec_reward, rec_rank, rec_list = self.recommend(unknown_facet)
            if not self.silence:
                print("Simulated Conversation Over: Success, Target {}/{}".format(rec_rank, len(rec_list)))
            return True, None, rec_reward
        if self.turn_count == self.turn_limit:
            if not self.silence:
                print("Simulated Conversation Over: Failed")
            return True, None, self.rq
        user_nl = self.user_turn(request_facet)
        self.get_dialogue_state()
        return False, self.dialogue_state, self.rc

    def user_turn(self, request_facet):
        user_nl = self.user.next_turn(request_facet)
        self.user_utt_list.append(user_nl)
        if not self.silence:
            print("Turn %d user: %s" % (self.turn_count, user_nl))
        return user_nl

    def get_dialogue_state(self):
        self.dialogue_state = self.bftracker.use_tracker_from_nl(self.user_utt_list, self.tracker_idx_list)

    def recommend(self, unknown_facet):
        business_list = self.rec.recommend_bussiness(self.user_name, self.dialogue_state, unknown_facet)
        business_list = business_list[:self.K]
        for rank_index, business_name in enumerate(business_list):
            rank_id = rank_index + 1
            if business_name == self.business_name:
                rec_reward = self.C * (self.K - rank_id + 1) / self.K
                return rec_reward, rank_id, business_list
        rec_reward = self.rq
        return rec_reward, -1, business_list