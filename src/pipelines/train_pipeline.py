
from src.utils.config import load_config
from src.utils.serialize import load_object
from src.components.ingestion import ingest_data
from src.components.transformation import transform_data
from src.components.train import train_model

from src.logger import logging
from src.exception import CustomException

def train(config):
    train_dataset, test_dataset = ingest_data(config)
    train_dataset, test_dataset = transform_data(train_dataset, test_dataset)
    accuracy = train_model(train_dataset, test_dataset, config)
    logging.info(f"Test accuracy: {accuracy}")
    print(f"Test accuracy: {accuracy}")
    return accuracy


if __name__=="__main__":
    config = config = load_config("configs/configs.yaml")
    accuracy = train(config)