"""Simple hyperparameter tuning in Ray using random redraws.

[1] - https://wwwjmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf.
"""
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
        "weight_decay",
        "decay_rate",
        "batch_norm",
        "resnet",
        "print_every",
    ],
)

possible_embed_sizes = np.arange(2, 5)
possible_batch_sizes = np.array([16, 32, 64, 128])
possible_lrs = np.array([1e-2, 5 * 1e-3, 1e-3, 5 * 1e-4, 1e-4, 5 * 1e-5, 1e-5])
possible_n_layers = np.array([1, 2, 3, 5, 8])
possible_hidden_sizes = np.array([8, 16, 32, 64])
possible_dropout_rates = np.array([0.1, 0.2, 0.3, 0.5,])
possible_decay_rates = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

GBDTArgs = namedtuple(
    "GBDTArgs",
    [
        "learning_rate",
        "num_rounds",
        "bagging_fraction",
        "lambda_l1",
        "max_depth",
        "early_stopping_round",
    ],
)

possible_early_stopping = np.array([3, 5, 10])
possible_gbdt_lrs = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
possible_num_rounds = np.array([500, 1000, 2000, 5000])
possible_max_depth = np.array([-1, 3, 5, 7])
possible_bagging_fraction = np.array([1.0, 0.9, 0.8, 0.7])
possible_lambda_l1 = np.array([0.0, 1e-5, 1e-4, 1e-3, 1e-2])


def random_params() -> Args:
    """Random set of hyperparameters for our regularized MLP model."""
    return Args(
        embed_size=np.random.choice(possible_embed_sizes),
        batch_size=np.random.choice(possible_batch_sizes),
        lr=np.random.choice(possible_lrs),
        n_layers=np.random.choice(possible_n_layers),
        n_epochs=5000,
        hidden_size=np.random.choice(possible_hidden_sizes),
        dropout_enabled=np.random.rand() >= 0.5,
        dropout_rate=np.random.choice(possible_dropout_rates),
        weight_decay=np.random.rand() >= 0.5,
        decay_rate=np.random.choice(possible_decay_rates),
        batch_norm=True,
        resnet=np.random.rand() >= 0.5,
        single_batch=False,
        print_every=None,
    )


def random_gbdt_params():
    """Random set of hyperparameters for a GBDT model."""
    return GBDTArgs(
        learning_rate=np.random.choice(possible_gbdt_lrs),
        num_rounds=np.random.choice(possible_num_rounds),
        bagging_fraction=np.random.choice(possible_bagging_fraction),
        lambda_l1=np.random.choice(possible_lambda_l1),
        max_depth=np.random.choice(possible_max_depth),
        early_stopping_round=np.random.choice(possible_early_stopping),
    )
