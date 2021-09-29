from typing import Sequence
import flax.linen as nn


def get_embedders(cardinalities, embedding_size):
    """Return Embed models for all columns to embed, they all map to an
    embedding space of the same size. We add 1 to the input cardinalities
    because we expect an unseen unknown token."""
    return [
        nn.Embed(num_embeddings=cardinality + 1, features=embedding_size)
        for cardinality in cardinalities
    ]


class MLP(nn.Module):
    layer_sizes: Sequence[int]
    dropout_rate: float = 0.0
    dropout: bool = False
    bias: float = 0.0

    @nn.compact
    def __call__(self, x, train=True):

        for layer_ix, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(layer_size)(x)
            if layer_ix != len(self.layer_sizes) - 1:
                x = nn.relu(x)
                if self.dropout:
                    x = nn.Dropout(
                        rate=self.dropout_rate, deterministic=not train
                    )(x)
            else:
                x = x + self.bias
        return x


class Resnet(nn.Module):
    layer_sizes: Sequence[int]
    dropout_rate: float = 0.0
    dropout: bool = False

    @nn.compact
    def __call__(self, x, train=True):
        x_last_shape = x.shape[-1]
        n_layers = len(self.layer_sizes)
        for (layer_ix, layer_size) in enumerate(self.layer_sizes):
            residual = x
            x = nn.Dense(layer_size)(x)
            if layer_ix != n_layers - 1:
                assert (
                    layer_size == x_last_shape
                ), f"incompatible residual shapes x:{x.shape}, layer:{layer_size}"
                x += residual
                x = nn.relu(x)
            if self.dropout:
                x = nn.Dropout(
                    rate=self.dropout_rate, deterministic=not train
                )(x)
        return x
