import numpy as np
from src.utils.serialize import load_object
from src.utils.config import load_config
from src.utils.prediction_utils import preprocess_image

# load once
config = load_config("configs/configs.yaml")
model_path = config["train"]["model_save_path"]
model = load_object(model_path)

class_names = ["Alpaca", "Not Alpaca"]

def prediction_pipeline(image):
    """
    image: PIL.Image.Image
    returns: string label
    """
    input_tensor = preprocess_image(image)
    preds = model(input_tensor, training=False)
    prediction = (preds.numpy() > 0.5).astype("int32").squeeze()
    print(prediction)

    return class_names[prediction]