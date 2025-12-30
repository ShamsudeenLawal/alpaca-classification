import os
import sys
import joblib
from src.exception import CustomException
from src.logger import logging
from tensorflow import keras

# object serialization
def save_object(file_path: str, obj) -> None:
    try:
        logging.info(f"Saving object to {file_path}...")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        if isinstance(obj, keras.Model):
            obj.save(file_path)
        else:
            with open(file_path, "wb") as file_obj:
                joblib.dump(obj, file_obj)
    except Exception as err:
        raise CustomException(err, sys)  # type: ignore

def load_object(file_path: str):
    try:
        logging.info(f"Loading object from {file_path}")
        if file_path.split(".")[-1] == "keras":
            loaded_object = keras.models.load_model(file_path)
        else:
            with open(file_path, "rb") as file_obj:
                loaded_object = joblib.load(file_obj)
        logging.info(f"Object successfully loaded.")
        
        return loaded_object
    
    except Exception as err:
        raise CustomException(err, sys) # type: ignore
