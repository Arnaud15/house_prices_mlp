from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import pytest
from flax.core.frozen_dict import unfreeze
from optax import sgd

from data_loader import get_dataset
from models import CustomMLP, init_params
from training_loop import train


def linear_data_helper(
    seed: int,
    n: int,
    p: int,
    bias: float,
    beta_scale: float = 1.0,
    noise_scale: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    exp_seed = random.PRNGKey(seed)
    exp_seed, out1_key, out2_key, out3_key = random.split(exp_seed, 4)

    X = random.normal(out1_key, shape=(n, p))
    noise = random.normal(out2_key, shape=(n,)) * noise_scale
    beta = 2 * beta_scale * (random.uniform(out3_key, shape=(p,)) - 0.5)
    y = X.dot(beta) + noise + bias
    return X, y


@pytest.fixture
def linear_data_fixture():
    return linear_data_helper(
        seed=123, n=100, p=10, bias=-2.0, beta_scale=1.0, noise_scale=1.0
    )


def test_dropout():
    """Test dropout regularization"""
    p = 7
    dummy_bsize = 32
    key_init, key_num, key_cat = random.split(random.PRNGKey(151543), 3)
    input_num = random.normal(key_num, shape=(dummy_bsize, p))
    input_cat = jnp.ones((dummy_bsize, 0))

    # Everything dropped
    custom_mlp = CustomMLP(
        layer_sizes=[1],
        vocab_sizes=[],
        embed_size=3,
        dropout_rate=1.0,
        dropout=True,
        bias=0.0,
    )

    params = init_params(
        key_init, custom_mlp, (dummy_bsize, p), (dummy_bsize, 0), dropout=True,
    )

    out = custom_mlp.apply(
        params, input_num, input_cat, rngs={"dropout": random.PRNGKey(123)}
    )
    assert jnp.all(out == 0.0)

    # Test mode
    out = custom_mlp.apply(
        params,
        input_num,
        input_cat,
        rngs={"dropout": random.PRNGKey(123)},
        train=False,
    )
    assert jnp.all(out != 0.0)

    # Equal when rate is 0.0
    custom_mlp = CustomMLP(
        layer_sizes=[1],
        vocab_sizes=[],
        embed_size=3,
        dropout_rate=0.0,
        dropout=True,
        bias=0.0,
    )
    out = custom_mlp.apply(
        params,
        input_num,
        input_cat,
        rngs={"dropout": random.PRNGKey(123)},
        train=True,
    )
    out_det = custom_mlp.apply(
        params,
        input_num,
        input_cat,
        rngs={"dropout": random.PRNGKey(123)},
        train=False,
    )
    assert jnp.allclose(out, out_det)

    # Equal in expectation
    custom_mlp = CustomMLP(
        layer_sizes=[1],
        vocab_sizes=[],
        embed_size=3,
        dropout_rate=0.5,
        dropout=True,
        bias=0.0,
    )
    new_input = random.normal(random.PRNGKey(123), shape=(1, p))
    concatd = jnp.concatenate([new_input for _ in range(10000)], axis=0)

    out_deterministic = custom_mlp.apply(
        params,
        concatd,
        input_cat,
        rngs={"dropout": random.PRNGKey(123)},
        train=False,
    )

    out_random = custom_mlp.apply(
        params,
        concatd,
        input_cat,
        rngs={"dropout": random.PRNGKey(123)},
        train=True,
    )
    assert jnp.abs(out_deterministic.mean() - out_random.mean()) < 1e-2


def test_resnet():
    """Test residual connections."""
    p = 7
    dummy_bsize = 3
    n_layers = 5
    key_init, key_num = random.split(random.PRNGKey(151543), 2)
    input_num = random.uniform(key_num, shape=(dummy_bsize, p))
    input_cat = jnp.ones((dummy_bsize, 0))

    custom_mlp = CustomMLP(
        layer_sizes=[p for _ in range(n_layers)],
        vocab_sizes=[],
        embed_size=3,
        dropout=False,
        batch_norm=False,
        bias=0.0,
        residuals=True,
    )

    params = init_params(
        key_init, custom_mlp, (dummy_bsize, p), (dummy_bsize, 0),
    )
    params = jax.tree_map(
        lambda x: jnp.identity(p) if len(x.shape) > 1 else x, params
    )

    out = custom_mlp.apply(params, input_num, input_cat)
    assert jnp.allclose(out, (2 ** n_layers) * input_num)


def test_batch_norm():
    """Test the batch norm"""
    p = 7
    dummy_bsize = 3
    many_train_steps = 500
    key_init, key_num1, key_num2 = random.split(random.PRNGKey(151543), 3)
    input_num = random.normal(key_num1, shape=(dummy_bsize, p))
    other_input_num = random.normal(key_num2, shape=(dummy_bsize, p))
    input_cat = jnp.ones((dummy_bsize, 0))

    custom_mlp = CustomMLP(
        layer_sizes=[p],
        vocab_sizes=[],
        embed_size=3,
        dropout=False,
        bias=0.0,
        batch_norm=True,
    )

    params = init_params(
        key_init, custom_mlp, (dummy_bsize, p), (dummy_bsize, 0),
    )
    params = jax.tree_map(
        lambda x: jnp.identity(p) if len(x.shape) > 1 else x, params
    )

    out, new_batch_stats = custom_mlp.apply(
        params, input_num, input_cat, train=True, mutable=["batch_stats"],
    )
    assert jnp.abs(out.mean()) < 1e-3
    assert jnp.abs(out.std() - 1.0) < 1e-3

    for _ in range(many_train_steps):
        out, new_batch_stats = custom_mlp.apply(
            {
                "params": params["params"],
                "batch_stats": new_batch_stats["batch_stats"],
            },
            input_num,
            input_cat,
            train=True,
            mutable=["batch_stats"],
        )
    out_trained = custom_mlp.apply(
        {
            "params": params["params"],
            "batch_stats": new_batch_stats["batch_stats"],
        },
        input_num,
        input_cat,
        train=False,
    )

    out_test = custom_mlp.apply(
        {
            "params": params["params"],
            "batch_stats": new_batch_stats["batch_stats"],
        },
        other_input_num,
        input_cat,
        train=False,
    )
    assert jnp.abs(out_trained.mean()) < 1e-1 * jnp.abs(out_test.mean())
    assert jnp.abs(out_trained.std() - 1.0) < 1e-1 * jnp.abs(
        out_test.std() - 1.0
    )


def test_bias():
    """Test bias initialization"""
    p = 13
    large_bias = -25.0
    dummy_bsize = 32
    tol = 1.0
    key_init, key_num, key_cat = random.split(random.PRNGKey(151543), 3)

    custom_mlp = CustomMLP(
        layer_sizes=[7, 1],
        vocab_sizes=[2],
        embed_size=5,
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


def test_seeding(linear_data_fixture):
    """Test seeding on a linear dataset"""
    X, y = linear_data_fixture
    seed = 1234

    linear_model = CustomMLP(
        layer_sizes=[3, 5, 1,],
        vocab_sizes=[],
        embed_size=1,
        dropout_rate=0.0,
        dropout=True,
        bias=y.mean(),
    )

    train_dataset = get_dataset(
        x_num=X,
        x_cat=jnp.ones((X.shape[0], 0)),
        y_data=y,
        batch_size=X.shape[0],
        buffer_size=X.shape[0],
        single_batch=True,
    )
    params, loss = train(
        random.PRNGKey(seed),
        model=linear_model,
        optimizer=sgd(1e-2),
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        num_epochs=100,
        num_input_shape=X.shape,
        cat_input_shape=(X.shape[0], 0),
        smoothing_alpha=1.0,
        hist_every=None,
        print_every=None,
    )
    params2, loss2 = train(
        random.PRNGKey(seed),
        model=linear_model,
        optimizer=sgd(1e-2),
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        num_epochs=100,
        num_input_shape=X.shape,
        cat_input_shape=(X.shape[0], 0),
        smoothing_alpha=1.0,
        hist_every=None,
        print_every=None,
    )
    assert jnp.abs(loss - loss2) < 1e-12
    bool_tree = jax.tree_multimap(
        lambda x, y: jnp.abs(x - y).mean() < 1e-12, params, params2
    )
    assert jax.tree_util.tree_reduce(
        lambda curr, x: curr and x, bool_tree, True
    )
