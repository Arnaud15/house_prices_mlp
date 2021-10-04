import numpy as np
import pandas as pd

from preprocess import *
from preprocess_utils import *


def with_nan(npoints: int) -> np.ndarray:
    arr = np.random.randn(npoints)
    nan_indexes = np.random.rand(npoints) >= 0.6
    arr[nan_indexes] = np.nan
    assert np.isnan(arr).sum() > 0
    return arr


def with_zeroes(npoints: int) -> np.ndarray:
    arr = np.random.randn(npoints)
    nan_indexes = np.random.rand(npoints) >= 0.6
    arr[nan_indexes] = 0.0
    assert (arr == 0).sum() > 0
    return arr


def test_preprocess_e2e():
    n_unique = 3
    n_repeats = 7
    offset = 5
    codes = np.arange(n_unique)
    repeated_codes = np.repeat(codes, n_repeats) + offset
    npoints = len(repeated_codes)
    y = np.random.rand(npoints)
    data = pd.DataFrame(
        {
            "ones": np.ones_like(repeated_codes),
            "zeros": np.zeros_like(repeated_codes),
            "unique": np.random.rand(npoints),
            "repeated": repeated_codes,
            "nan_values": with_nan(npoints),
            "zero_values": with_zeroes(npoints),
            "SalePrice": y,
        }
    )
    encoded_transforms = ["embed", "isnan", "is0"]
    numeric_transforms = ["lognorm", "identity", "scale"]
    to_encode = []
    to_numeric = []
    for col in data.columns:
        if col == "SalePrice":
            continue
        for transform in encoded_transforms:
            to_encode.append((col, transform))
        if col == "nan_values":
            continue
        for transform in numeric_transforms:
            if (
                col == "zeros" or col == "zero_values"
            ) and transform == "lognorm":
                continue
            to_numeric.append((col, transform))
    X_train_numeric, y_train, numeric_info = preprocess_train(data, to_numeric)
    X_train_encoding, _, encoding_info = preprocess_train(data, to_encode)
    X_eval_numeric, y_eval, _ = preprocess_eval(data, numeric_info)
    X_eval_encoding, _, _ = preprocess_eval(data, encoding_info)
    assert np.allclose(X_train_numeric, X_eval_numeric)
    assert np.allclose(y_train, y_eval)
    assert np.all(X_train_encoding == X_eval_encoding)


def test_scale():
    n_unique = 100
    vals = np.random.randn(n_unique)
    scaled, minmax = scale_column(vals)
    assert minmax is not None
    min_, max_ = minmax
    # check that min and max of the scaled vector are -1 and 1
    assert np.abs(scaled.min() + 1.0) <= FLOAT_TOL
    assert np.abs(scaled.max() - 1.0) <= FLOAT_TOL
    # check that saved min and max are min and max of original vector
    assert np.abs(min_ - vals.min()) <= FLOAT_TOL
    assert np.abs(max_ - vals.max()) <= FLOAT_TOL

    vals_large_range = np.random.randn(n_unique) * 10

    # random vals below min should be scaled below - 1
    below = scale_from_minmax(
        vals_large_range[vals_large_range <= min_], minmax
    )
    assert np.all(below <= -1.0)
    # random vals above max should be scaled above 1
    above = scale_from_minmax(
        vals_large_range[vals_large_range >= max_], minmax
    )
    assert np.all(above >= 1.0)
    # random vals in between should be in -1, 1
    between = scale_from_minmax(
        vals_large_range[
            (vals_large_range >= min_) * (vals_large_range <= max_)
        ],
        minmax,
    )
    assert np.all((between >= -1.0) * (between <= 1.0))

    # (max + min) / 2 should be scaled to 0
    zeros = scale_from_minmax(np.array([(max_ + min_) / 2.0]), minmax)
    assert np.allclose(zeros, np.array([0.0]))

    # identical values should be scaled to 0.0
    identical = np.ones(n_unique)
    zeros, should_be_none = scale_column(identical)
    assert np.allclose(zeros, np.zeros_like(zeros))
    assert should_be_none is None

    # if info is None, all values should be zeroed out
    zeros = scale_from_minmax(vals, None)
    assert np.allclose(zeros, np.zeros_like(zeros))


def test_encode_from_info():
    np.random.seed(1515)
    n_unique = 100
    rd_vals = np.random.randn(n_unique)
    embedded, mapping = encode_column(rd_vals)
    embedded_again = encode_from_mapping(rd_vals, mapping)
    assert np.all(embedded == embedded_again)

    rd_vals_2 = np.random.randn(n_unique)
    embedded_unk = encode_from_mapping(rd_vals_2, mapping)
    assert np.all(embedded_unk == n_unique)


def test_encode_column():
    n_unique = 7
    n_repeats = 3
    offset = 5
    repeated_codes = np.arange(n_unique)
    already_encoded = np.repeat(repeated_codes, n_repeats)
    encoded, mapping = encode_column(already_encoded + offset,)
    assert np.all(already_encoded == encoded)
    assert len(mapping) == n_unique + 1
    assert mapping["UNK_TOKEN"] == n_unique

    np.random.seed(15)
    n_unique = 100
    all_unique = np.random.randn(n_unique)
    encoded, mapping = encode_column(all_unique,)
    to_sorted = np.argsort(encoded)
    assert np.all(encoded[to_sorted] == np.arange(all_unique.shape[0]))
    assert np.allclose(
        sorted(
            [x for x in mapping.keys() if x != "UNK_TOKEN"],
            key=lambda x: mapping[x],
        ),
        all_unique[to_sorted],
    )
    assert mapping["UNK_TOKEN"] == n_unique
