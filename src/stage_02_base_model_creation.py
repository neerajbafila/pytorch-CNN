import os
import logging
import re
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d

from src.utils.common import read_yaml, create_directories

STAGE = "stage_02_base_model_creation"

logging.basicConfig(filename=os.path.join("logs", 'running_logs.log'),
                format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
                filemode='a'
                )

class CNN(nn.Module):
    def __init__(self, in_, out_):
        super(CNN, self).__init__()

        self.conv_pool_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_, out_channels=8, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_pool_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Flatten = nn.Flatten()
        self.FC_01 = nn.Linear(in_features=16*4*4, out_features=128)
        self.FC_02 = nn.Linear(in_features=128, out_features=64)
        self.FC_03 = nn.Linear(in_features=64, out_features=out_)
    
    def forward_pass(self, x):
        x = self.conv_pool_1(x)
        x = self.conv_pool_2(x)
        x = self.Flatten(x)
        x = self.FC_01(x)
        x = ReLU(x)
        x = self.FC_02(x)
        x = ReLU(x)
        x = self.FC_03(x)

        return x

