from typing import Tuple

import jax.numpy as jnp
import jax.random as random
import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataset(
    x_data, y_data, batch_size: int, buffer_size: int, numpy=False,
):
    data = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    data = data.batch(batch_size)
    data = data.shuffle(buffer_size=buffer_size)
    if numpy:
        return tfds.as_numpy(data)
    else:
        return data


def linear_data(
    seed: int,
    n: int,
    p: int,
    bias: float,
    beta_scale: float = 1.0,
    noise_scale: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    exp_seed = random.PRNGKey(seed)
    exp_seed, out1_key, out2_key, out3_key = random.split(exp_seed, 4)

    X = random.normal(out1_key, shape=(n, p))
    noise = random.normal(out2_key, shape=(n,)) * noise_scale
    beta = 2 * beta_scale * (random.uniform(out3_key, shape=(p,)) - 0.5)
    Y = X.dot(beta) + noise + bias
    return X, Y


def train_test_split(
    dataset: tf.data.Dataset, test_share: float
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Split a tf dataset into train and test"""
    n_eval_batches = int(test_share * len(dataset))
    test = dataset.take(n_eval_batches)
    train = dataset.skip(n_eval_batches)
    return train, test

