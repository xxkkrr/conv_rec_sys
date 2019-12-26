import sys
import os
import argparse
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
from usersim.UserReal import UserReal
from belieftracker.BeliefTrackerModule import BeliefTrackerModule
from belieftracker.BeliefTrackerConfig import BeliefTrackerConfig
import agents

print(agents.AgentRLConfig.__file__)

parser = argparse.ArgumentParser()
parser.add_argument("user", choices=[0, 1], help="0 for simulate user, 1 for real user", type=int)
parser.add_argument("agent", choices=[0, 1], help="0 for rule agent, 1 for RL agent", type=int)
parser.add_argument("--num", help="conversation times", type=int, default=1)

args = parser.parse_args()

rec = RecModule(RecModuleConfig())

if args.agent == 0:
    agent = AgentRule(AgentRuleConfig())
if args.agent == 1:
    agent = AgentRL(AgentRLConfig())
    agent.load_model(False)

if args.user == 0:
    user = UserSim(UserSimConfig())
if args.user == 1:
    user = UserReal(UserSimConfig())

bftracker = BeliefTrackerModule(BeliefTrackerConfig(), False)
bftracker.load_all_model()
DM = DialogueManager(DialogueManagerConfig(), rec, agent, user, bftracker)

example_list = []
with open('./data/example.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        example_list.append(row)

TestNum = args.num
for Test in range(TestNum):
    print("---------------------------")
    print("Conversation {}:".format(Test))
    data = random.choice(example_list)
    print("UserName: {}, BussinessName: {}".format(data[0], data[1]))
    DM.initialize_episode(data[0], data[1])
    IsOver = False
    while not IsOver:
        IsOver, reward, rank_id, business_list = DM.next_turn()
    print(business_list)
    if rank_id == -1:
        print("Conversation Over: Failed")
    else:
        print("Conversation Over: Success, Target {}/{}".format(rank_id, len(business_list)))
