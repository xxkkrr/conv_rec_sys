import sys
import os
GRANDFA = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(GRANDFA)
from FMConfig import FMConfig
from FMModule import FMModule

fm_module = FMModule(FMConfig(), True)
fm_module.train_model()
fm_module.eva_model()