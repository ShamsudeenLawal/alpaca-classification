import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

def transform_data(train_dataset, test_dataset):
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset, test_dataset
