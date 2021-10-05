from typing import Optional

import jax


def squared_loss(y, predicted):
    """
    Simple elementwise squared loss.
    """
    return (y - predicted) ** 2


mse_loss = jax.vmap(squared_loss, in_axes=0)


def update_running(obs: float, loss: Optional[float], decay: float):
    """
    Small helper to update an ewma with current value loss, where loss is None
    at initialization.
    """
    assert decay >= 0.0 and decay <= 1.0
    if loss is None:
        return obs
    else:
        return obs * (1.0 - decay) + loss * decay
