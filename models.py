from typing import Any, List, Tuple
import flax.linen as nn
import jax.numpy as jnp
import jax.random as random

PRNG = jnp.ndarray
Pytree = Any

def init_params(
    rng: PRNG,
    num_input_shape: Tuple[int, ...],
    cat_input_shape: Tuple[int, ...],
    dropout: bool=False,
) -> Pytree:
    """ Initializes pytree parameters for our CustomMLP."""
    if dropout:
        num_input_key, cat_input_key, params_key, dropout_key = random.split(rng, num=4)
    else:
        num_input_key, cat_input_key, params_key = random.split(rng, num=3)

    num_input = random.normal(num_input_key, shape=num_input_shape)
    cat_input = random.bernoulli(cat_input_key, shape=cat_input_shape)
    base_keys = {"params": params_key}
    if dropout:
        base_keys.update(
            {"dropout": dropout_key}
        )
    return CustomMLP.init(base_keys, num_input, cat_input)


class CustomMLP(nn.Module):
    layer_sizes: List[int]
    embedding_sizes: List[Tuple[int, int]]
    dropout_rate: float = 0.0
    dropout: bool = False
    bias: float = 0.0

    @nn.compact
    def __call__(self, x_numeric, x_categorical, train=True):
        assert len(self.embedding_sizes) == x_categorical.shape[1]
        dense = [x_numeric]
        for (embed_ix, (vocab_size, embed_size)) in enumerate(
            self.embedding_sizes
        ):
            dense.append(
                nn.Embed(vocab_size, embed_size)(x_categorical[:, embed_ix])
            )
        x = jnp.concatenate(dense)

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
    layer_sizes: List[int]
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
