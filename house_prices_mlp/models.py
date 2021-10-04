from typing import Any, List, Tuple

import flax.linen as nn
import jax.numpy as jnp
import jax.random as random

PRNG = jnp.ndarray
Pytree = Any


def init_params(
    rng: PRNG,
    custom_mlp: nn.Module,
    num_input_shape: Tuple[int, ...],
    cat_input_shape: Tuple[int, ...],
    dropout: bool = False,
) -> Pytree:
    """ Initializes pytree parameters for our CustomMLP."""
    if dropout:
        num_input_key, cat_input_key, params_key, dropout_key = random.split(
            rng, num=4
        )
    else:
        num_input_key, cat_input_key, params_key = random.split(rng, num=3)

    num_input = random.normal(num_input_key, shape=num_input_shape)
    cat_input = random.bernoulli(cat_input_key, shape=cat_input_shape).astype(
        int
    )
    base_keys = {"params": params_key}
    if dropout:
        base_keys.update({"dropout": dropout_key})
    batch_stats = None
    return custom_mlp.init(base_keys, num_input, cat_input)


class CustomMLP(nn.Module):
    layer_sizes: List[int]
    vocab_sizes: List[int]
    embed_size: int
    dropout_rate: float = 0.0
    dropout: bool = False
    bias: float = 0.0
    batch_norm: bool = False
    residuals: bool = False

    @nn.compact
    def __call__(self, x_numeric, x_categorical, train=True):
        """Forward method for our custom MLP. Assumes x_numeric and x_cat are dim >=2 arrays."""
        assert len(self.vocab_sizes) == x_categorical.shape[-1]
        dense = [x_numeric]
        for (embed_ix, vocab_size) in enumerate(self.vocab_sizes):
            dense.append(
                nn.Embed(vocab_size, self.embed_size)(
                    x_categorical[:, embed_ix]
                )
            )
        x = jnp.concatenate(dense, axis=1)
        assert (
            x.shape[-1]
            == x_numeric.shape[-1] + x_categorical.shape[-1] * self.embed_size
        )

        for layer_ix, layer_size in enumerate(self.layer_sizes):
            start = x
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)
            if self.dropout:
                x = nn.Dropout(
                    rate=self.dropout_rate, deterministic=not train
                )(x)
            x = nn.Dense(layer_size)(x)
            if layer_ix != len(self.layer_sizes) - 1:
                x = nn.relu(x)
            else:
                x = x + self.bias
            if self.residuals and x.shape == start.shape:
                x = x + start
        return x
