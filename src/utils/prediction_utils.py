import io
import numpy as np
from PIL import Image
import tensorflow as tf
from src.utils.config import load_config
from fastapi import File, UploadFile, HTTPException

config = load_config("configs/configs.yaml")
image_size = config["data"]["image_size"]

def load_image_from_upload(file: UploadFile) -> Image.Image:
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file format.")
    
    # read image as a stream of bytes
    image_bytes = file.file.read()
    image_stream = io.BytesIO(image_bytes)

    # start the stream from the beginning
    image_stream.seek(0)
    
    # load the image using PIL
    image = Image.open(image_stream).convert("RGB")

    return image

def preprocess_image(image: Image.Image) -> tf.Tensor:
    image = image.resize((image_size, image_size))
    image_array = np.array(image, dtype="float32")
    image_tensor = tf.expand_dims(image_array, axis=0)
    return tf.convert_to_tensor(image_tensor, dtype=tf.float32)

