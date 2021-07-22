import tensorflow as tf
import tensorflow_datasets as tfds

def get_dataset(x_data, y_data, batch_size):
    data = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    data = data.shuffle(buffer_size=1000)
    data = data.batch(batch_size)
    return tfds.as_numpy(data)
