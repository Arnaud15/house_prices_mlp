"""Preprocessing utilities, transforming the raw house prices csv data to
cleaner numerical and categorical features, using specified input transforms
for selected columns of the csv."""
from collections import namedtuple
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .preprocess_utils import *

TransformInfo = Tuple[str, str, Any]  # column, transform name, optional data

Datapoints = namedtuple("Datapoints", ["y", "X_num", "X_cat"])

NUMERIC_TRANSFORMS = {"scale", "lognorm", "identity"}
CATEGORICAL_TRANSFORMS = {"isnan", "is0", "embed"}
MIN_CARDINALITY = 50


def get_transformed_data(
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    test_data: pd.DataFrame,
    transforms: List[Tuple[str, str]],
) -> Tuple[Datapoints, Datapoints, Datapoints, List[int]]:
    numeric_t = [
        (col, transfo)
        for (col, transfo) in transforms
        if transfo in NUMERIC_TRANSFORMS
    ]
    X_num, y, numeric_info = preprocess_train(train_data, numeric_t)
    cat_t = [
        (col, transfo)
        for (col, transfo) in transforms
        if transfo in CATEGORICAL_TRANSFORMS
    ]
    X_cat, _, cat_info = preprocess_train(train_data, cat_t)

    retained_cat_indices = [
        ix
        for ix in range(X_cat.shape[1])
        if np.unique(X_cat[:, ix], return_counts=True)[1].min()
        >= MIN_CARDINALITY
    ]
    X_cat = X_cat[:, retained_cat_indices]
    cardinalities = [
        len(np.unique(X_cat[:, ix])) for ix in range(X_cat.shape[1])
    ]

    X_num_eval, y_eval, nan_to_rm_eval = preprocess_eval(
        eval_data, numeric_info
    )
    X_cat_eval, _, _ = preprocess_eval(eval_data, cat_info)
    X_cat_eval = X_cat_eval[:, retained_cat_indices]

    X_num_test, _, nan_to_rm_test = preprocess_eval(test_data, numeric_info)
    X_cat_test, _, _ = preprocess_eval(test_data, cat_info)
    X_cat_test = X_cat_test[:, retained_cat_indices]

    total_nan_ixs = set(nan_to_rm_eval + nan_to_rm_test)
    kept_ixs = [ix for ix in range(X_num.shape[1]) if ix not in total_nan_ixs]
    train_data = Datapoints(y=y, X_num=X_num[:, kept_ixs], X_cat=X_cat)
    eval_data = Datapoints(
        y=y_eval, X_num=X_num_eval[:, kept_ixs], X_cat=X_cat_eval
    )
    test_data = Datapoints(
        y=None, X_num=X_num_test[:, kept_ixs], X_cat=X_cat_test
    )
    return train_data, eval_data, test_data, cardinalities


def preprocess_train(
    data: pd.DataFrame, transforms: List[Tuple[str, str]],
) -> Tuple[np.ndarray, np.ndarray, List[TransformInfo]]:
    """Transform training data and return preprocessed data (X, y) and
    preprocessing parameters.
    Args:
        - data: training data from the house prices dataset.
        - transforms: list of (col_name, transform_name) specifying which
          transforms to apply to selected columns.
        Possible transforms are
        - embed: columns that are encoded.
        - scale: columns that are scaled in [-1, 1]
        - lognorm: columns that are log normalized.
        - identity: columns that are used as-is.
        - isnan: transformed into the binary indicator x is nan.
        - is0: transformed into the binary indicator x == 0.
    Returns:
        (X, y, info)
        - X of shape (n_rows, n_columns) the transformed training data.
        - y the target vector of shape (n_rows,).
        - info a dictionary that maps a column to transform info
    """
    # Extract the response vector
    assert "SalePrice" in data.columns, "incomplete frame"
    y_vec = np.log(
        data.loc[:, "SalePrice"].values
    )  # see eda notebook - should be log normalized

    # Transform other columns according to retained transforms
    transformed = []
    preprocess_info = []
    for (col, transform) in transforms:
        assert (
            col != "SalePrice"
        ), "forbidden to use the response vector as a feature!"
        if transform == "embed":
            encoded, mapping = encode_column(
                data.loc[:, col].astype(str).values,
            )
            transformed.append(encoded)
            preprocess_info.append((col, transform, mapping))
        elif transform == "scale":
            scaled, min_max = scale_column(data.loc[:, col].values)
            transformed.append(scaled)
            preprocess_info.append((col, transform, min_max))
        elif transform == "lognorm":
            scaled, mean_std = log_std(data.loc[:, col].values)
            transformed.append(scaled)
            preprocess_info.append((col, transform, mean_std))
        elif transform == "identity":
            transformed.append(data.loc[:, col].values)
            preprocess_info.append((col, transform, None))
        elif transform == "isnan":
            transformed.append(np.isnan(data.loc[:, col].values))
            preprocess_info.append((col, transform, None))
        elif transform == "is0":
            transformed.append(data.loc[:, col].values == 0)
            preprocess_info.append((col, transform, None))
        else:
            raise NotImplementedError(
                f"tranform {transform} is not implemented."
            )
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
    data: pd.DataFrame, preprocess_info: List[TransformInfo],
) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
    """Transform evaluation data using preprocessing information from training data.
    Args:
        - data: the evaluation data to transform
        - preprocess_info: For each column to transform, the information
          required to transform it in the same way that previous training data
          has been transformed.
    Returns:
        (X, y) evaluation data transformed and ready to be processed."""
    transformed = []
    nan_to_remove_ixs = []
    for (ix, (col, transform, info)) in enumerate(preprocess_info):
        assert (
            col != "SalePrice"
        ), "forbidden to use the response vector as a feature!"
        if transform == "embed":
            encoded = encode_from_mapping(
                data.loc[:, col].astype(str).values, info
            )
            transformed.append(encoded)
        elif transform == "scale":
            scaled = scale_from_minmax(data.loc[:, col].values, info)
            if (
                np.isnan(scaled).sum() > 0
            ):  # some nan values can appear outside of train
                nan_to_remove_ixs.append(ix)
            transformed.append(scaled)
        elif transform == "lognorm":
            log_transformed = log_std_from_stats(data.loc[:, col].values, info)
            if np.isnan(log_transformed).sum() > 0:
                nan_to_remove_ixs.append(ix)
            transformed.append(log_transformed)
        elif transform == "identity":
            transformed.append(data.loc[:, col].values)
        elif transform == "isnan":
            transformed.append(np.isnan(data.loc[:, col].values))
        elif transform == "is0":
            transformed.append(data.loc[:, col].values == 0)
        else:
            raise NotImplementedError(
                f"tranform {transform} is not implemented."
            )
    assert len(transformed) == len(preprocess_info)
    design_matrix = np.concatenate(
        [arr.reshape(-1, 1) for arr in transformed], axis=1
    )
    assert len(design_matrix.shape) == 2
    assert design_matrix.shape == (len(data), len(transformed))
    if "SalePrice" in data.columns:
        y_vec = np.log(data.loc[:, "SalePrice"].values)
        assert y_vec.shape[0] == design_matrix.shape[0]
    else:
        y_vec = None
    return design_matrix, y_vec, nan_to_remove_ixs
