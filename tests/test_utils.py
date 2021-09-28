import pytest

from models import MLP, Resnet
from utils import init_params

import jax.random as random
import jax.numpy as jnp


@pytest.fixture(params=["MLP", "Resnet"])
def model_init(request):
    if request.param == "MLP":
        return MLP([4, 7, 1], dropout=False), 3
    elif request.param == "Resnet":
        return Resnet([13, 13, 13, 1], dropout=False), 4
    else:
        raise NotImplementedError


def test_init_params(model_init):
    """Test that bias setting works and produces outputs with the correct mean"""
    p = 13
    large_bias = -25.0
    dummy_bsize = 32
    tol = 1.0
    key_init, key_input = random.split(random.PRNGKey(151543))

    params = init_params(
        key_init,
        model_init[0],
        (p,),
        extra_keys=[],
        bias=large_bias,
        layer_name=str(model_init[1] - 1),
    )

    input_arr = random.normal(key_input, (dummy_bsize, p))
    out = model_init[0].apply(params, input_arr)

    dev = (out.mean(0) - large_bias) / large_bias
    assert jnp.abs(dev) < tol, dev
