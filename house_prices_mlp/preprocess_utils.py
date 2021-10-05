"""Preprocessing utilities for the house prices dataset."""
# TODO: should we standardize the log normalized columns? I would think so.
# add the is0 and isNa transforms
# tuples issue?
# N/A sanity checks
# embeddings should be separated before called by MLP
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

FLOAT_TOL = 1e-12

EmbedMapping = Dict[float, int]


def encode_column(col: np.ndarray) -> Tuple[np.ndarray, EmbedMapping]:
    """Turns a numpy array into an integer array corresponding to its unique
    values.
    Args:
        col: the numpy array to encode
    Returns:
        (encoded_col, mapping)
        - encoded_col has shape col.shape. It contains the integer codes
          corresponding to each value in the input array
        - mapping is a dictionary with n_uniques keys. keys are observed values
          in the column, their corresponding value is their embedding indice.
    """
    assert len(col.shape) == 1, "only embedding 1D arrays"
    _, indices, encoded_col = np.unique(
        col, return_index=True, return_inverse=True, return_counts=False
    )
    n_uniques = indices.shape[0]
    mapping = {col[indice]: idx for (idx, indice) in enumerate(indices)}
    mapping["UNK_TOKEN"] = n_uniques  # add unknown token
    return (
        encoded_col,
        mapping,
    )


def encode_from_mapping(col: np.ndarray, mapping: EmbedMapping) -> np.ndarray:
    """Embeds a columns from an existing mapping from floats to codes."""
    assert "UNK_TOKEN" in mapping
    encoded = np.empty_like(col)
    for (ix, val) in enumerate(col):
        try:
            encoded[ix] = mapping[val]
        except KeyError:
            encoded[ix] = mapping["UNK_TOKEN"]
    return encoded


def scale_column(
    col: np.ndarray,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    """Scale entries in the input to the range [-1, 1].
    Args:
        col: the ndarray to scale
    Returns:
        scaled: the scaled output
        min_max: the min and the max values observed and used for scaling
    """
    assert len(col.shape) == 1
    assert not np.any(np.isnan(col)), "NaN values in array to rescale"
    min_, max_ = col.min(), col.max()
    if (max_ - min_) < FLOAT_TOL:
        return np.zeros_like(col), None
    else:
        return 2 * (col - min_) / (max_ - min_) - 1, (min_, max_)


def scale_from_minmax(
    col: np.ndarray, minmax: Optional[Tuple[float, float]]
) -> np.ndarray:
    """Scale entries in the input using existing min max values"""
    if minmax is None:
        return np.zeros_like(col)
    else:
        min_, max_ = minmax
        assert max_ - min_ >= FLOAT_TOL
        return 2 * (col - min_) / (max_ - min_) - 1


def log_std(col: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Normalize inputs using a log, then standardize them"""
    assert col.min() > 0.0
    log_transformed = np.log(col)
    mean_, std_ = log_transformed.mean(), log_transformed.std()
    return (log_transformed - mean_) / (std_ + FLOAT_TOL), (mean_, std_)


def log_std_from_stats(
    col: np.ndarray, mean_std: Tuple[float, float]
) -> np.ndarray:
    """Normalize inputs using a log, a mean offset and a std scale."""
    mean_, std_ = mean_std
    assert std_ >= 0.0
    assert col.min() > 0.0
    return (np.log(col) - mean_) / (std_ + FLOAT_TOL)
