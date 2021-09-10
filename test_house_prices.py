import numpy as np

from house_prices import embed_column


def test_embed_column():
    n_unique = 7
    n_repeats = 3
    offset = 5
    repeated_codes = np.arange(n_unique)
    already_encoded = np.repeat(repeated_codes, n_repeats)
    encoded, mapping = embed_column(already_encoded + offset, add_unkown_token=False)
    assert np.all(already_encoded == encoded)
    assert len(mapping) == n_unique
    _, mapping_unk = embed_column(already_encoded, add_unkown_token=True)
    assert mapping_unk["UNK_TOKEN"] == n_unique

    np.random.seed(15)
    n_unique = 100
    all_unique = np.random.randn(n_unique)
    encoded, mapping = embed_column(all_unique, add_unkown_token=False)
    to_sorted = np.argsort(encoded)
    assert np.all(encoded[to_sorted] == np.arange(all_unique.shape[0]))
    assert np.allclose(sorted(list(mapping.keys()), key=lambda x: mapping[x]), all_unique[to_sorted])
    assert "UNK_TOKEN" not in mapping
    _, mapping_unk = embed_column(all_unique, add_unkown_token=True)
    assert mapping_unk["UNK_TOKEN"] == n_unique

