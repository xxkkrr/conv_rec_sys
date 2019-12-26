import pickle
import sys
import os
GRANDFA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(GRANDFA)
from recommendersystem.RecModule import RecModule
from recommendersystem.RecModuleConfig import RecModuleConfig
from agents.AgentRule import AgentRule
from agents.AgentRuleConfig import AgentRuleConfig
from usersim.UserSim import UserSim
from usersim.UserSimConfig import UserSimConfig
from belieftracker.BeliefTrackerModule import BeliefTrackerModule
from belieftracker.BeliefTrackerConfig import BeliefTrackerConfig
from tqdm import tqdm

class DialogueManager:
    def __init__(self, rec, agent, user, bftracker):
        self.tracker_idx_list = [0, 1, 2, 3, 4]
        self.facet_action_dict = {"categories":0, "state":1, "city":2, "price":3, "stars":4, "recommend":5}
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
        #print("Turn %d agent: %s" % (self.turn_count, agent_nl))
        return request_facet, agent_nl

    def user_turn(self, request_facet):
        user_nl = self.user.next_turn(request_facet)
        self.user_utt_list.append(user_nl)
        #print("Turn %d user: %s" % (self.turn_count, user_nl))
        return user_nl

    def get_dialogue_state(self):
        self.dialogue_state = self.bftracker.use_tracker_from_nl(self.user_utt_list, self.tracker_idx_list)

    # def recommend(self):
    #     business_list = self.rec.recommend_bussiness(self.user_name, self.dialogue_state, self.agent.current_unknown_facet())
    #     business_list = business_list[:self.K]
    #     for rank_index, business_name in enumerate(business_list):
    #         rank_id = rank_index + 1
    #         if business_name == self.business_name:
    #             rec_reward = self.C * (self.K - rank_id + 1) / self.K
    #             return rec_reward, rank_id, business_list
    #     rec_reward = self.rq
    #     return rec_reward, -1, business_list

    def next_turn(self):
        request_facet, agent_nl = self.agent_turn()
        agent_action = self.facet_action_dict[request_facet]
        if agent_action == 5:
            return True, self.dialogue_state.tolist(), agent_action
        user_nl = self.user_turn(request_facet)
        dialogue_state = self.dialogue_state
        self.get_dialogue_state()
        if type(dialogue_state) != type(None):
            return False, dialogue_state.tolist(), agent_action
        else:
            return False, None, None

with open('RL_data.pkl','rb') as f:
    train_data, dev_data, test_data = pickle.load(f)

rec = RecModule(RecModuleConfig())
agent = AgentRule(AgentRuleConfig())
user = UserSim(UserSimConfig())
bftracker = BeliefTrackerModule(BeliefTrackerConfig(), True)
bftracker.load_all_model()
DM = DialogueManager(rec, agent, user, bftracker)


dialogue_state_list = []
agent_action_list = []

for data in tqdm(train_data, ncols=10):
    #print("UserName: {}, BussinessName: {}".format(data[0], data[1]))
    DM.initialize_episode(data[0], data[1])
    IsOver = False
    while not IsOver:
        IsOver, dialogue_state, agent_action = DM.next_turn()
        if dialogue_state != None:
            dialogue_state_list.append(dialogue_state)
            agent_action_list.append(agent_action)

# print(dialogue_state_list)
# print(agent_action_list)

with open('RL_pretrain_data.pkl','wb') as f:
    pickle.dump([dialogue_state_list, agent_action_list], f)