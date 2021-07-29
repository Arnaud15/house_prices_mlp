from typing import Optional, Tuple
import os

from flax.core.frozen_dict import FrozenDict, unfreeze
import flax.linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import jax.random as random
import optax

from board import SummaryWriter
from utils import *


def train_step(
    rng, params, opt_state, x, y, model, optimizer, loss_fn, metric_fn,
):
    @jax.jit
    def pure_loss(params):
        predicted = model.apply(params, x, rngs={"dropout": rng})
        loss_items = loss_fn(y, predicted)
        return jnp.mean(loss_items), predicted
        # weight_decay(
        #     scale=l2_scale, params_tree=params
        # )  # TODO no good reason to compute weight decay by default

    (loss_step, predicted_step), grad_step = jax.value_and_grad(
        pure_loss, has_aux=True
    )(params)
    param_updates, opt_state = optimizer.update(grad_step, opt_state)
    metric_step = jnp.sqrt(jnp.mean(metric_fn(y, predicted_step)))
    params = optax.apply_updates(params, param_updates)

    return params, opt_state, loss_step, metric_step


def init_params(
    rng,
    model: nn.Module,
    inputs_shape: Tuple[int],
    bias: Optional[float] = None,
    layer_name: str = "",
):
    rng, rd_input_key, params_key, dropout_key = random.split(rng, num=4)
    dummy_input = random.normal(rd_input_key, shape=inputs_shape)
    params = model.init(
        {"params": params_key, "dropout": dropout_key}, dummy_input
    )

    if bias is None:
        return params

    assert isinstance(bias, float)
    dico = {"/".join(k): v for k, v in flatten_dict(unfreeze(params)).items()}
    new_dico = {
        tuple(key.split("/")): value + bias
        if ("bias" in key) and (layer_name in key)
        else value
        for key, value in dico.items()
    }
    return FrozenDict(unflatten_dict(new_dico))


def train(
    rng,
    model,
    optimizer,
    dataset,
    loss_fn,
    metric_fn,
    num_epochs,
    inputs_shape,
    layer_name,
    bias,
    output_dir=None,
    hist_every=None,
    print_every=None,
):
    if output_dir is not None:
        assert isinstance(output_dir, str)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            print(f"Directory '{output_dir}' did not exist and was created.")
        writer = SummaryWriter(output_dir)
    if hist_every is not None:
        assert output_dir is not None
        assert isinstance(hist_every, int)
    if print_every is not None:
        assert output_dir is not None
        assert isinstance(print_every, int)

    rng, init_params_rng = random.split(rng, num=2)
    params = init_params(
        rng=init_params_rng,
        model=model,
        inputs_shape=inputs_shape,
        bias=bias,
        layer_name=layer_name,
    )

    opt_state = optimizer.init(params)

    step = 0
    for _ in range(num_epochs):
        epoch_running_loss = None
        for (x_train, y_train) in dataset:
            rng, rng_step = random.split(rng, num=2)
            params, opt_state, loss_step, metric_step = train_step(
                rng_step,
                params,
                opt_state,
                x_train,
                y_train,
                model,
                optimizer,
                loss_fn,
                metric_fn,
            )
            epoch_running_loss = update_running(
                obs=loss_step, loss=epoch_running_loss, decay=0.9
            )
            step += 1
            if (print_every is not None) and (step % print_every == 0):
                print(
                    f"Step {step}, Metric: {metric_step:.3f}, Loss: {loss_step:.3f}"
                )
            if (hist_every is not None) and (step % print_every == 0):
                writer.scalar("running_loss", epoch_running_loss, step=step)
    return params
