import sys
import os
GRANDFA = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(GRANDFA)
import random
import csv
import pickle
from tqdm import tqdm
from dialoguemanager.DialogueManager import DialogueManager
from dialoguemanager.DialogueManagerConfig import DialogueManagerConfig
from recommendersystem.RecModule import RecModule
from recommendersystem.RecModuleConfig import RecModuleConfig
from agents.AgentRule import AgentRule
from agents.AgentRuleConfig import AgentRuleConfig
from agents.AgentRL import AgentRL
from agents.AgentRLConfig import AgentRLConfig
from usersim.UserSim import UserSim
from usersim.UserSimConfig import UserSimConfig
from belieftracker.BeliefTrackerModule import BeliefTrackerModule
from belieftracker.BeliefTrackerConfig import BeliefTrackerConfig

rec = RecModule(RecModuleConfig())
#agent = AgentRule(AgentRuleConfig())
agent = AgentRL(AgentRLConfig())
agent.load_model(False)
user = UserSim(UserSimConfig())
bftracker = BeliefTrackerModule(BeliefTrackerConfig(), True)
bftracker.load_all_model()
DM = DialogueManager(DialogueManagerConfig(), rec, agent, user, bftracker)

with open('../data/RL_data/RL_data.pkl','rb') as f:
    train_data, dev_data, test_data = pickle.load(f)
data_list = test_data

# TestNum = 10
# for Test in range(TestNum):
#     print("---------------------------")
#     print("Simulated Conversation {}:".format(Test))
#     data = random.choice(data_list)
#     print("UserName: {}, BussinessName: {}".format(data[0], data[1]))
#     DM.initialize_episode(data[0], data[1])
#     IsOver = False
#     while not IsOver:
#         IsOver, reward, rank_id, business_list = DM.next_turn()
#     print(business_list)
#     if rank_id == -1:
#         print("Simulated Conversation Over: Failed")
#     else:
#         print("Simulated Conversation Over: Success, Target {}/{}".format(rank_id, len(business_list)))

all_count = 0
success_count = 0
rank_total = 0
rank_all_count = 0
reward_total = 0
turn_count_sum = 0
for data in tqdm(data_list, ncols=10):
    all_count += 1
    DM.initialize_episode(data[0], data[1])
    IsOver = False
    turn_count = 0
    while not IsOver:
        turn_count += 1
        IsOver, reward, rank_id, business_list = DM.next_turn()
    if rank_id != -1:
        rank_total += rank_id
        rank_all_count += 1
        success_count += 1
    reward_total += reward
    turn_count_sum += turn_count

print("mean reward: ", float(reward_total)/all_count)
print("mean turn: ", float(turn_count_sum)/all_count)
print("success rate:", float(success_count)/all_count)
print("mean rank: ", float(rank_total)/rank_all_count)


