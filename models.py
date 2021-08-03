from typing import Sequence
import flax.linen as nn


class MLP(nn.Module):
    layer_sizes: Sequence[int]
    dropout: bool = False

    @nn.compact
    def __call__(self, x):

        for layer_ix, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(layer_size)(x)
            if layer_ix != len(self.layer_sizes) - 1:
                x = nn.relu(x)
                dropout_rate = 0.5
            else:
                dropout_rate = 0.2
            if self.dropout:
                x = nn.Dropout(deterministic=False, rate=dropout_rate)(x)
        return x


class Resnet(nn.Module):
    layer_sizes: Sequence[int]
    dropout: bool = False

    @nn.compact
    def __call__(self, x, train=True):
        x_last_shape = x.shape[-1]
        n_layers = len(self.layer_sizes)
        for (layer_ix, layer_size) in enumerate(self.layer_sizes):
            residual = x
            x = nn.Dense(layer_size)(x)
            if layer_ix == n_layers - 1:
                dropout_rate = 0.2
            else:
                assert (
                    layer_size == x_last_shape
                ), f"incompatible residual shapes x:{x.shape}, layer:{layer_size}"
                x += residual
                x = nn.relu(x)
                dropout_rate = 0.5
            x = nn.Dropout(
                dropout_rate, deterministic=not train or not self.dropout
            )(x)
        return x
