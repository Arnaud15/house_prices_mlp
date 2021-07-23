import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataset(x_data, y_data, batch_size, single_batch=False):
    data = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    data = data.batch(batch_size)
    if single_batch:
        data = data.take(1)
    data = data.shuffle(buffer_size=1000)
    return tfds.as_numpy(data)
