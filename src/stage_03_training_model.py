from calendar import EPOCH
from cgi import test
import logging
import os
import torch
from tqdm import tqdm
import torch.nn as nn
# import torch.nn as nn
import argparse
from src import stage_01_get_data
from src.stage_02_base_model_creation import CNN
from src.utils.common import read_yaml, create_directories

logging.basicConfig(filename=os.path.join("logs", "running_logs.log"),
                    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
                    level=logging.INFO,
                    filemode="a"
                    )


def main(config_path):
    try:
        DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
        content = read_yaml(config_path)
        base_model_path = os.path.join(content['artifacts']['model'], content['artifacts']['base_model'])
        logging.info(f"loading model {base_model_path}")
        loaded_model = torch.load(base_model_path)
        loaded_model.eval()
        logging.info(f'{base_model_path} model loaded')
        # load model in cuda
        loaded_model.to(DEVICE)
        logging.info(f"{loaded_model} is loaded in {DEVICE}")
        LR = content['params']['LR']
        optimizer = torch.optim.Adam(loaded_model.parameters(), lr=LR) # optimizer
        train_data_loader, test_data_loader, label_map = stage_01_get_data.main(config_path)
        EPOCHS = content['Epoch']
        criterion = nn.CrossEntropyLoss() # loss function
        for epoch in range(EPOCHS):
            with tqdm(train_data_loader) as tqdm_epoch:
                for image, label in tqdm_epoch:
                    tqdm_epoch.set_description(f"EPOCH{epoch+1} / {EPOCH}")
                    #put image in cuda
                    image = image.to(DEVICE)
                    label = label.to(DEVICE)

                    #forward pass
                    outputs = loaded_model(image)
                    loss = criterion(outputs, label)
                    
                    # backward pro
                    optimizer.zero_grad() # clear past grad
                    loss.backward() # calculate gradient
                    optimizer.step() # update the weight
                    tqdm_epoch.set_postfix(loss=loss.item())
        logging.info(f"Model trained successfully")
        trained_model_path = os.path.join(content['artifacts']['model'], content['artifacts']['trained_model'])
        torch.save(loaded_model, trained_model_path)
        logging.info(f'trained model saved at {trained_model_path}')
    except Exception as e:
        logging.exception(e)
        print(e)


if __name__ == "__main__":
    try:
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("--config", "-c", default="config/config.yaml")
        parsed_arg = arg_parser.parse_args()
        main(parsed_arg.config)
    except Exception as e:
        print(e)