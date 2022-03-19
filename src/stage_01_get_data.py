import logging
import os
import argparse
from random import shuffle
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets



from src.utils.common import create_directories, read_yaml


STAGE = "stage_01_get_data"

log_file = os.path.join("logs", 'running_logs.log')


logging.basicConfig(filename=log_file, level=logging.INFO,
                    format= "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
                    filemode='a'

                    )

def main(config_path):
    content = read_yaml(config_path)
    data_folder_path = content['Data']['root_data_folder']
    create_directories([data_folder_path])
    logging.info(f"getting data")
    train_data = datasets.FashionMNIST(root=data_folder_path, train=True, download=True,
                                        transform=transforms.ToTensor())
    test_data = datasets.FashionMNIST(root=data_folder_path, train=False, download=True, transform= transforms.ToTensor())
    logging.info(f"data is available at {data_folder_path}")

    logging.info(f"getting dataloader")
    train_data_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=content['params']['BATCH_SIZE'])
    test_data_loader = DataLoader(dataset=test_data, batch_size=content['params']['BATCH_SIZE'], shuffle=False)

    return train_data, test_data

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = parser.parse_args()
    try:
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} started<<<<<<<<<<<<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} completed<<<<<<<<<<<<<<<")
    except Exception as e:
        print(e)
        logging.exception(e)
        raise e

