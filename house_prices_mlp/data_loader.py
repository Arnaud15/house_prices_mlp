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
    if single_batch:
        data = data.take(1)
    else:
        data = data.shuffle(buffer_size=buffer_size)
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
