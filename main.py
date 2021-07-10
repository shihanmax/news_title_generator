import json
import logging
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append("..")
from title_gen.config import Config
from title_gen.trainer import Trainer

torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO)


Trainer(config=Config).start_train()
