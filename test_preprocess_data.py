import numpy as np

from house_prices import embed_column


def test_embed_from_info():
    np.random.seed(1515)
    n_unique = 100
    rd_vals = np.random.randn(n_unique)
    embedded, mapping = embed_column(rd_vals)
    embedded_again = embed_from_mapping(rd_vals, mapping)
    assert np.all(embedded == embedded_again)

    rd_vals_2 = np.random.randn(n_unique)
    embedded_unk = embed_from_mapping(rd_vals_2, mapping)
    assert np.all(embedded_unk == n_unique)


def test_embed_column():
    n_unique = 7
    n_repeats = 3
    offset = 5
    repeated_codes = np.arange(n_unique)
    already_encoded = np.repeat(repeated_codes, n_repeats)
    encoded, mapping = embed_column(already_encoded + offset,)
    assert np.all(already_encoded == encoded)
    assert len(mapping) == n_unique + 1
    assert mapping["UNK_TOKEN"] == n_unique

    np.random.seed(15)
    n_unique = 100
    all_unique = np.random.randn(n_unique)
    encoded, mapping = embed_column(all_unique,)
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

