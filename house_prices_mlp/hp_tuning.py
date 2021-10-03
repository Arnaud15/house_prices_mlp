from collections import namedtuple

import numpy as np

Args = namedtuple(
    "Args",
    [
        "embed_size",
        "batch_size",
        "lr",
        "n_layers",
        "n_epochs",
        "hidden_size",
        "dropout_enabled",
        "dropout_rate",
        "single_batch",
    ],
)

possible_embed_sizes = np.arange(1, 5)
possible_batch_sizes = np.array([16, 32, 64, 128])
possible_lrs = np.array([1e-2, 5 * 1e-3, 1e-3, 5 * 1e-4, 1e-4, 5 * 1e-5, 1e-5])
possible_n_layers = np.arange(1, 6)
possible_hidden_sizes = np.array([8, 16, 32, 64])
possible_dropout_rates = np.array([0.0, 0.1, 0.2, 0.3, 0.5,])


def random_params() -> Args:
    return Args(
        embed_size=np.random.choice(possible_embed_sizes),
        batch_size=np.random.choice(possible_batch_sizes),
        lr=np.random.choice(possible_lrs),
        n_layers=np.random.choice(possible_n_layers),
        n_epochs=1000,
        hidden_size=np.random.choice(possible_hidden_sizes),
        dropout_enabled=True,
        dropout_rate=np.random.choice(possible_dropout_rates),
        single_batch=False,
    )
