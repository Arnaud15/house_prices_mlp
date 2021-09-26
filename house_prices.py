"""Preprocessing utilities for the house prices dataset."""
# TODO: should we standardize the log normalized columns? I would think so.
from typing import Any, Dict, List, Optional, Tuple


import numpy as np
import pandas as pd


FLOAT_TOL = 1e-12

EmbedMapping = Dict[float, int]

def embed_column(col: np.ndarray) -> Tuple[np.ndarray, EmbedMapping]:
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


def embed_from_mapping(col: np.ndarray, mapping: EmbedMapping) -> np.ndarray:
    """Embeds a columns from an existing mapping from floats to codes."""
    assert "UNK_TOKEN" in mapping
    embedded = np.empty_like(col)
    for (ix, val) in enumerate(col):
        try:
            embedded[ix] = mapping[val]
        except KeyError:
            embedded[ix] = mapping["UNK_TOKEN"]
    return embedded


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


def scale_from_minmax(col: np.ndarray, minmax: Optional[Tuple[float, float]]) -> np.ndarray:
    """Scale entries in the input using existing min max values"""
    if minmax is None:
        return np.zeros_like(col)
    else:
        min_, max_ = minmax
        assert max_ - min_ >= FLOAT_TOL
        return 2 * (col - min_) / (max_ - min_) - 1


def log_normalize(col: np.ndarray) -> np.ndarray:
    """Normalize inputs using a log."""
    assert col.min() > 0.0
    return np.log(col)


TransformInfo = Tuple[str, str, Any] # column, transform name and optional data

def preprocess_train(
    data: pd.DataFrame,
    transforms: List[Tuple[str, str]],
) -> Tuple[np.ndarray, np.ndarray, List[TransformInfo]]:
    """Transform training data and return preprocessed data (X, y) and
    preprocessing parameters.
    Args:
        - data: training data from the house prices dataset.
        - transforms: list of (col_name, transform_name) specifying which
          transforms to apply to selected columns.
        Possible transforms are
        - embed: columns that are embedded.
        - scale: columns that are scaled in [-1, 1]
        - lognorm: columns that are log normalized.
        - identity: columns that are used as-is.
    Returns:
        (X, y, info)
        - X of shape (n_rows, n_columns) the transformed training data.
        - y the target vector of shape (n_rows,).
        - info a dictionary that maps a column to transform info
    """
    # Extract the response vector
    assert "SalePrice" in data.columns, "incomplete frame"
    y_vec = data.loc[:, "SalePrice"].values

    # Transform other columns according to retained transforms
    transformed = []
    preprocess_info = []
    for (col, transform) in transforms:
        assert col != "SalePrice", "forbidden to use the response vector as a feature!"
        if transform == "embed":
            embedded, mapping = embed_column(
                data.loc[:, col].values,
            )
            transformed.append(embedded)
            preprocess_info.append((col, transform, mapping))
        elif transform == "scale":
            scaled, min_max = scale_column(data.loc[:, col].values)
            transformed.append(scaled)
            preprocess_info.append((col, transform, min_max))
        elif transform == "lognorm":
            transformed.append(log_normalize(data.loc[:, col].values))
            preprocess_info.append((col, transform, None))
        elif transform == "identity":
            transformed.append(data.loc[:, col].values)
            preprocess_info.append((col, transform, None))
        else:
            raise NotImplementedError(f"tranform {transform} is not implemented.")
    assert len(transformed) == len(preprocess_info)
    assert len(transformed) == len(transforms)

    # Extract the design matrix and return
    design_matrix = np.concatenate(
        [arr.reshape(-1, 1) for arr in transformed], axis=1
    )
    assert len(design_matrix.shape) == 2
    assert design_matrix.shape == (len(data), len(transformed))
    assert y_vec.shape[0] == design_matrix.shape[0]
    return design_matrix, y_vec, preprocess_info


def preprocess_eval(
    data: pd.DataFrame,
    preprocess_info: List[TransformInfo],
) -> np.ndarray:
    """Transform evaluation data using preprocessing information from training data.
    Args:
        - data: the evaluation data to transform
        - preprocess_info: For each column to transform, the information
          required to transform it in the same way that previous training data
          has been transformed.
    Returns:
        (X, y) evaluation data transformed and ready to be processed."""
    transformed = []
    for (col, transform, info) in preprocess_info:
        assert col != "SalePrice", "forbidden to use the response vector as a feature!"
        if transform == "embed":
            embedded = embed_from_mapping(
                data.loc[:, col].values,
                info
            )
            transformed.append(embedded)
        elif transform == "scale":
            scaled= scale_from_minmax(data.loc[:, col].values, info)
            transformed.append(scaled)
        elif transform == "lognorm":
            transformed.append(log_normalize(data.loc[:, col].values))
        elif transform == "identity":
            transformed.append(data.loc[:, col].values)
        else:
            raise NotImplementedError(f"tranform {transform} is not implemented.")
    assert len(transformed) == len(preprocess_info)
    design_matrix = np.concatenate(
        [arr.reshape(-1, 1) for arr in transformed], axis=1
    )
    assert len(design_matrix.shape) == 2
    assert design_matrix.shape == (len(data), len(transformed))
    assert "SalePrice" in data.columns, "incomplete frame"
    y_vec = data.loc[:, "SalePrice"].values
    assert y_vec.shape[0] == design_matrix.shape[0]
    return design_matrix, y_vec

