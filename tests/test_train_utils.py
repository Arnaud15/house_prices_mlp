import jax.numpy as jnp
import jax.random as random

from house_prices_mlp.models import CustomMLP, init_params


def test_init_params():
    """Test that bias setting works and produces outputs with the correct mean"""
    p = 13
    large_bias = -25.0
    dummy_bsize = 32
    tol = 1.0
    key_init, key_num, key_cat = random.split(random.PRNGKey(151543), 3)

    custom_mlp = CustomMLP(
        layer_sizes=[7, 1],
        embedding_sizes=[(2, 5)],
        dropout_rate=0.0,
        dropout=False,
        bias=large_bias,
    )

    params = init_params(
        key_init,
        custom_mlp,
        (dummy_bsize, p),
        (dummy_bsize, 1),
        dropout=False,
    )

    input_num = random.normal(key_num, shape=(dummy_bsize, p))
    input_cat = random.bernoulli(key_cat, shape=(dummy_bsize, 1)).astype(int)
    out = custom_mlp.apply(params, input_num, input_cat)

    dev = (out.mean(0) - large_bias) / large_bias
    assert jnp.abs(dev) < tol, dev
