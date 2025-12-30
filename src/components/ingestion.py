import sys
import tensorflow as tf
from tensorflow import keras

from src.logger import logging
from src.exception import CustomException

def ingest_data(config):
    try:
        logging.info("Data ingestion started...")
        data_config = config["data"]
        train_dataset = keras.preprocessing.image_dataset_from_directory(
            directory=data_config["data_directory"],
            shuffle=data_config["shuffle"],
            batch_size=data_config["batch_size"],
            image_size=(data_config["image_size"], data_config["image_size"]),
            validation_split=data_config["validation_split"],
            seed=data_config["seed"],
            subset="training"
        )

        validation_dataset = keras.preprocessing.image_dataset_from_directory(
            directory=data_config["data_directory"],
            shuffle=data_config["shuffle"],
            batch_size=data_config["batch_size"],
            image_size=(data_config["image_size"], data_config["image_size"]),
            validation_split=data_config["validation_split"],
            seed=data_config["seed"],
            subset="validation"
            
        )
        logging.info("Data ingested successfully...")
        return train_dataset, validation_dataset
    except Exception as err:
        raise CustomException(err, sys) # type: ignore
