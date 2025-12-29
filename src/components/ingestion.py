import tensorflow as tf
from tensorflow import keras

from src.logger import logging
from src.exception import CustomException

def ingest_data(config):
    try:
        logging.info("Data ingestion started...")
        train_dataset = keras.preprocessing.image_dataset_from_directory(
            directory=config["data_directory"]
            shuffle=config["shuffle"],
            batch_size=config["batch_size"],
            image_size=(config["image_size"], config["image_size"]),
            validation_split=config["validation_split"],
            seed=config["seed"]
            subset="training"
        )

        validation_dataset = keras.preprocessing.image_dataset_from_directory(
            directory=config["data_directory"]
            shuffle=config["shuffle"],
            batch_size=config["batch_size"],
            image_size=(config["image_size"], config["image_size"]),
            validation_split=config["validation_split"],
            seed=config["seed"]
            subset="validation"
            
        )
        logging.info("Data ingested successfully...")
        return train_dataset, validation_dataset
    except Exception as err:
        raise CustomException(err, sys) # type: ignore
