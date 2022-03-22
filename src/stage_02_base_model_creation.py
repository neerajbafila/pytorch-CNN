import os
import logging
from numpy import full
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool2d
import argparse
from src.utils.common import read_yaml, create_directories

STAGE = "stage_02_base_model_creation"

logging.basicConfig(filename=os.path.join("logs", "running_logs.log"),
                    level=logging.INFO,
                    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
                    filemode="a+"
                    )


class CNN(nn.Module):
    def __init__(self, in_=1, out_=10):
        super(CNN, self).__init__()
        logging.info("making base model........")
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
        logging.info("base model created")

    def forward(self, x):
        logging.info("Making forwardPass....")
        x = self.conv_pool_1(x)
        x = self.conv_pool_2(x)
        x = self.Flatten(x)
        x = self.FC_01(x)
        x = F.relu(x)
        x = self.FC_02(x)
        x = F.relu(x)
        x = self.FC_03(x)
        logging.info(f"forwardPass done ")
        return x

if __name__ == "__main__":
    try:
        logging.info("\n************************************")
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} started<<<<<<<<<<<<<<<")
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("--config", '-c', default="config/config.yaml")
        parsed_args = arg_parser.parse_args()
        content  = read_yaml(parsed_args.config)
        model_path = os.path.join(content['artifacts']['model'])
        create_directories([model_path])
        model_name = content['artifacts']['base_model']
        full_model_path = os.path.join(model_path, model_name)
        model_ob = CNN()
        torch.save(model_ob, full_model_path)
        logging.info(f"model created and saved at {full_model_path}")
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} completed<<<<<<<<<<<<<<<")
        logging.info("\n************************************")
        
    except Exception as e:
        logging.exception(e)
        raise e