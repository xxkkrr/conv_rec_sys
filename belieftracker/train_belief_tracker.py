import sys
import os
GRANDFA = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(GRANDFA)
from BeliefTrackerConfig import BeliefTrackerConfig
from BeliefTrackerModule import BeliefTrackerModule

btc = BeliefTrackerConfig()
btm = BeliefTrackerModule(btc, True)
for i in range(1,5):
    btm.train_model(i)
    btm.eva_model(i)