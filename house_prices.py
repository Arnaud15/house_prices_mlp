from typing import Any, Dict, List, Optional, Tuple


import numpy as np
import pandas as pd


def embed_column(
    col: np.ndarray, add_unkown_token: bool
) -> Tuple[np.ndarray, Dict]:
    """Turns a numpy array into an integer array corresponding to its unique
    values.
    Args:
        col: the numpy array to encode
        add_unkown_token: True if an additional integer embedding is added.
    Returns:
        (encoded_col, mapping)
        encoded_col has shape col.shape = the integer codes corresponding to
        each value in the input array
        mapping is a dictionary with n_uniques keys. keys are observed values
        in the column, their corresponding value is their embedding indice.
    """
    assert len(col.shape) == 1, "only embedding 1D arrays"
    _, indices, encoded_col = np.unique(
        col, return_index=True, return_inverse=True, return_counts=False
    )
    n_uniques = indices.shape[0]
    mapping = {col[indice]: idx for (idx, indice) in enumerate(indices)}
    if add_unkown_token:
        mapping["UNK_TOKEN"] = n_uniques
    return (
        encoded_col,
        mapping,
    )


def scale_min_max(val, min_val, max_val):
    """Dummy utility to scale values between -1 and 1"""
    return 2 * (val - min_val) / (max_val - min_val) - 1


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
    if (max_ - min_) < 1e-12:
        return np.zeros_like(col), None
    return scale_min_max(col, min_, max_), (min_, max_)


def log_normalize(col: np.ndarray) -> np.ndarray:
    """Normalize inputs using a log"""
    assert col.min() > 0.0
    return np.log(col)


def flatten_preprocessed(
    transformed_cols: Dict[str, np.ndarray], info: Dict[str, Any]
) -> Tuple[np.ndarray, List[Any]]:
    """Utility to flatten a dictionary of transformed columns to a matrix of
    training examples, and their associated dictionary of preprocessing info to a list."""
    assert len(transformed_cols) == len(info)
    all_cols = sorted(list(transformed_cols.keys()))
    design_matrix = np.concatenate(
        [transformed_cols[col].reshape(-1, 1) for col in all_cols], axis=1
    )
    assert len(design_matrix.shape) == 2
    assert design_matrix.shape[1] == len(all_cols)
    return design_matrix, [info[col] for col in all_cols]


def preprocess_data(
    data: pd.DataFrame,
    to_embed: List[str],
    to_scale: List[str],
    to_lognorm: List[str],
    to_identity: List[str],
) -> Tuple[np.ndarray, List[Any]]:
    assert "SalePrice" in data.columns, "incomplete frame"
    transformed = {}
    preprocess_info = {}

    for col in to_embed:
        embedded, mapping = embed_column(
            data.loc[:, col].values, add_unkown_token=True
        )
        transformed[col] = embedded
        preprocess_info[col] = mapping

    for col in to_scale:
        scaled, min_max = scale_column(data.loc[:, col].values)
        transformed[col] = scaled
        preprocess_info[col] = min_max

    for col in to_lognorm:
        transformed[col] = log_normalize(data.loc[:, col].values)
        preprocess_info[col] = None

    for col in to_identity:
        transformed[col] = data.loc[:, col].values
        preprocess_info[col] = None

    design_matrix, info = flatten_preprocessed(transformed, preprocess_info)
    assert design_matrix.shape[0] == len(data)
    return design_matrix, info

