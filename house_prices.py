from typing import List, Tuple


import numpy as np
import pandas as pd


def embed_column(col: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Turns a numpy array into an integer array corresponding to its unique values
    Args:
        col: the numpy array to encode
    Returns:
        (encoded_col, mapping)
        encoded_col has shape col.shape = the integer codes corresponding to each value in the input array
        mapping has shape (n_uniques,) and values mapping[i] = value corresponding to code i
    """
    assert len(col.shape) == 1, "only embedding 1D arrays"
    _, indices, encoded_col = np.unique(
        col, return_index=True, return_inverse=True, return_counts=False
    )
    return encoded_col, col[indices]


def preprocess_data(
    data: pd.DataFrame,
    to_embed: List[str],
    to_scale: List[str],
    to_lognorm: List[str],
    to_drop: List[str],
) -> pd.DataFrame:
    assert "SalePrice" in data.columns, "incomplete frame"
    data = data.drop(columns=to_drop)
    return data

