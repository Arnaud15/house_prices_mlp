from collections import namedtuple
from typing import Tuple

import numpy as np
import ray

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
    ],
)

possible_embed_sizes = np.arange(1, 5)
possible_batch_sizes = np.array([16, 32, 64, 128])
possible_lrs = np.array([1e-2, 5 * 1e-3, 1e-3, 5 * 1e-4, 1e-4, 5 * 1e-5, 1e-5])
possible_n_layers = np.arange(1, 6)
possible_hidden_sizes = np.array([8, 16, 32, 64])
possible_dropout_rates = np.array([0.0, 0.1, 0.2, 0.3, 0.5,])
possible_decay_rates = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])


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
        weight_decay=np.random.rand() >= 0.5,
        decay_rate=np.random.choice(possible_decay_rates),
        batch_norm=True,  # always for now
        resnet=np.random.rand() >= 0.5,
    )


def hp_search_ray(n_repeats: int) -> Tuple[np.ndarray, str]:
    ray.init(num_cpus=5)

    @ray.remote
    def closure():
        args = random_params()
        return house_prices_train(args), args

    out = ray.get([closure.remote() for _ in range(n_repeats)])
    return sorted(out, key=lambda x: x[0][0])[0]
