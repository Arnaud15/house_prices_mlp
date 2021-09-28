from typing import Any, Optional, Tuple, List

import jax
import jax.numpy as jnp
import jax.random as random

from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import flax.linen as nn

Pytree = Any
PRNG = jnp.ndarray


def poisson_loss(y, predicted):
    """
    Poisson NLL, parameterized so that `predicted` can take any scalar value (can be negative).
    """
    return -y * predicted + jnp.exp(predicted)


def squared_loss(y, predicted):
    """
    Simple elementwise squared loss.
    """
    return (y - predicted) ** 2


mse_loss = jax.vmap(jax.jit(squared_loss), in_axes=0)


def weight_decay(scale, params_tree):
    """
    Weight decay in JAX.
    """
    l2_sum = jax.tree_util.tree_reduce(
        lambda x, y: x + jnp.sum(y ** 2), params_tree, initializer=0.0
    )
    dim = jax.tree_util.tree_reduce(
        lambda x, y: x + jnp.prod(jnp.array(y.shape)),
        params_tree,
        initializer=0.0,
    )
    return scale * l2_sum / dim


def update_running(obs: float, loss: Optional[float], decay: float):
    """
    Small helper to update an ewma with current value loss, where loss is None
    at initialization.
    """
    assert decay > 0.0 and decay <= 1.0
    if loss is None:
        return obs
    else:
        return obs * (1.0 - decay) + loss * decay


def init_params(
    rng: PRNG,
    model: nn.Module,
    inputs_shape: Tuple[int],
    extra_keys: List[str] = [],
    bias: Optional[float] = None,
    layer_name: str = "",
) -> Pytree:
    """ Initializes pytree parameters of a deep learning model from `flax.linen`.

    Args:
        rng (jnp.ndarray of size 2): PRNG key
        model
        inputs_shape: shape of inputs that the model should be able to process.
        used to initialize model parameters.
        extra_keys: list of extra keys that might be necessary to update model parameters (e.g ["dropout"]).
        bias: optional bias to initialize in the model.
        layer_name: name of the layer to target to initialize the bias.
    Returns:
        params: a pytree of initialized model parameters.
    """
    n_extra_keys = len(extra_keys)
    out_keys = random.split(rng, num=3 + n_extra_keys)
    if n_extra_keys:
        rng, rd_input_key, params_key, extra_keys_rngs = (
            out_keys[0],
            out_keys[1],
            out_keys[2],
            out_keys[3:],
        )
    else:
        rng, rd_input_key, params_key = out_keys
        extra_keys_rngs = []
    dummy_input = random.normal(rd_input_key, shape=inputs_shape)
    base_keys = {"params": params_key}
    base_keys.update(
        {name: key for name, key in zip(extra_keys, extra_keys_rngs)}
    )
    params = model.init(base_keys, dummy_input)

    if bias is None:
        return params

    # Add bias to the model in the specified layer
    assert isinstance(bias, float)
    assert isinstance(layer_name, str)
    assert layer_name
    dico = {"/".join(k): v for k, v in flatten_dict(unfreeze(params)).items()}
    new_dico = {
        tuple(key.split("/")): value + bias
        if ("bias" in key) and (layer_name in key)
        else value
        for key, value in dico.items()
    }
    return FrozenDict(unflatten_dict(new_dico))
