import numpy as np

from house_prices import embed_column


def test_embed_column():
    repeated_codes = np.arange(10)
    already_encoded = np.repeat(repeated_codes, 3)
    encoded, mapping = embed_column(already_encoded)
    assert np.all(already_encoded == encoded)
    assert np.all(mapping == repeated_codes)

    np.random.seed(15)
    all_unique = np.random.randn(100)
    encoded, mapping = embed_column(all_unique)
    to_sorted = np.argsort(encoded)
    assert np.all(encoded[to_sorted] == np.arange(all_unique.shape[0]))
    assert np.allclose(mapping, all_unique[to_sorted])
