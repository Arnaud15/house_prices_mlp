from typing import Tuple

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataset(
    x_num,
    x_cat,
    y_data,
    batch_size: int,
    buffer_size: int,
    single_batch: bool = False,
):
    data = tf.data.Dataset.from_tensor_slices((x_num, x_cat, y_data))
    data = data.batch(batch_size)
    data = data.shuffle(buffer_size=buffer_size)
    if single_batch:
        data = data.take(1)
    return tfds.as_numpy(data)


def train_test_split_pandas(
    dataframe: pd.DataFrame, test_share: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train and test."""
    assert test_share >= 0.0 and test_share <= 1.0
    train_indices = np.random.rand(len(dataframe)) >= test_share
    train_data, eval_data = (
        dataframe[train_indices],
        dataframe[~train_indices],
    )
    assert len(train_data) + len(eval_data) == len(dataframe)
    return train_data, eval_data


def train_test_split_tf(
    dataset: tf.data.Dataset, test_share: float = 0.3
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Split a tf dataset into train and test."""
    n_eval_batches = int(test_share * len(dataset))
    test = dataset.take(n_eval_batches)
    train = dataset.skip(n_eval_batches)
    return train, test


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
