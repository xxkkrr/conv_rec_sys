import sys
import os
GRANDFA = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(GRANDFA)
from AgentRL import AgentRL
from AgentRLConfig import AgentRLConfig

agent = AgentRL(AgentRLConfig())
agent.pretrain_train()
agent.pretrain_eva()