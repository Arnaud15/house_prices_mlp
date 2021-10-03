from typing import Optional

import jax
import jax.numpy as jnp


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


mse_loss = jax.vmap(squared_loss, in_axes=0)


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
