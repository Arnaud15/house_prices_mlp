from collections import namedtuple

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


def random_params() -> Args:
    pass
