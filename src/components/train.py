import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.exception import CustomException
from src.logger import logging
from src.utils.serialize import save_object

# augmentation model
def data_augmenter():
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2)
    ])

    return data_augmentation

def build_model(config):
    try:
        logging.info("Model building started")
        data_config = config["data"]
        train_config = config["train"]

        IMAGE_SIZE = data_config["image_size"]
        IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
        
        # load components
        preprocessor = tf.keras.applications.mobilenet_v2.preprocess_input
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMAGE_SHAPE,
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False
        data_augmentation = data_augmenter()
        
        # assemble model
        inputs = keras.Input(shape=IMAGE_SHAPE)
        x = data_augmentation(inputs)
        x = preprocessor(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(train_config["dropout"])(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # compile model
        optimizer = keras.optimizers.get(train_config["optimizer"])
        optimizer.learning_rate = train_config["learning_rate"]
        loss=keras.losses.get(train_config["loss"])

        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=train_config["metrics"]
        )
        logging.info("Model building completed")

        return model
    
    except Exception as e:
        raise CustomException(e, sys)
    

def train_model(train_dataset, test_dataset, config):
    model = build_model(config)
    try:
        logging.info("Model training started")
        train_config = config["train"]
        if train_config["early_stopping"]:
            callback = keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=10, mode="max",
                restore_best_weights=True
                )

        # train model        
        history = model.fit(
            train_dataset,
            epochs=train_config["epochs"],
            validation_data=test_dataset,
            callbacks=[callback] if train_config["early_stopping"] else None
        )
        
        logging.info("Model training completed")
        
        # save model
        logging.info("Saving model")
        save_object(train_config["model_save_path"], model)
        logging.info(f"Model saved at {train_config['model_save_path']}")

        # evaluate model
        logging.info("Evaluating model")
        eval_result = model.evaluate(test_dataset)
        test_accuracy = eval_result[1]
        logging.info(f"Test accuracy: {test_accuracy}")

        return test_accuracy
    
    except Exception as e:
        raise CustomException(e, sys)
    
