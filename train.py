"""
Simple training module in Flax.

TODO: train state to jit
TODO: checkpointing
"""

import os

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

from board import SummaryWriter
from utils import init_params, update_running


def train_step(
    rng, params, opt_state, x, y, model, optimizer, loss_fn,
):
    @jax.jit
    def pure_loss(params):
        predicted = model.apply(
            params, x, rngs={"dropout": rng}
        )  # TODO generalize this rngs arg
        loss_items = loss_fn(y, predicted)
        return jnp.mean(loss_items), predicted

    (loss_step, predicted_step), grad_step = jax.value_and_grad(
        pure_loss, has_aux=True
    )(params)
    param_updates, opt_state = optimizer.update(grad_step, opt_state)
    params = optax.apply_updates(params, param_updates)

    residuals_step = y - predicted_step
    return params, opt_state, loss_step, residuals_step


def eval_step(
    rng, params, x, y, model, loss_fn,
):
    @jax.jit  # TODO look into this
    def pure_loss(params):
        predicted = model.apply(
            params, x, rngs={"dropout": rng}
        )  # TODO generalize this rngs arg
        loss_items = loss_fn(y, predicted)
        return jnp.mean(loss_items), predicted

    loss_step, predicted_step = pure_loss(params)
    residuals_step = y - predicted_step
    return loss_step, residuals_step


def train(
    rng,
    model,
    optimizer,
    train_dataset,
    eval_dataset,
    loss_fn,
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
        running_train_loss = None
        running_eval_loss = None
        for ((x_train, y_train), (x_eval, y_eval)) in tfds.as_numpy(
            tf.data.Dataset.zip((train_dataset, eval_dataset))
        ):
            rng, rng_train, rng_eval = random.split(rng, num=3)
            params, opt_state, loss_step, res_step = train_step(
                rng=rng_train,
                params=params,
                opt_state=opt_state,
                x=x_train,
                y=y_train,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
            )
            loss_eval_step, _ = eval_step(
                rng_eval,
                params,
                x=x_eval,
                y=y_eval,
                model=model,
                loss_fn=loss_fn,
            )
            running_train_loss = update_running(
                obs=loss_step,
                loss=running_train_loss,
                decay=0.9,  # TODO make this an argument
            )
            running_eval_loss = update_running(
                obs=loss_eval_step, loss=running_eval_loss, decay=0.9
            )
            step += 1
            if (print_every is not None) and (step % print_every == 0):
                writer.scalar("train_loss", running_train_loss, step=step)
                writer.scalar("validation_loss", running_eval_loss, step=step)
                print(f"Step {step} | Loss: {loss_step:.3f}")
            if (hist_every is not None) and (step % hist_every == 0):
                writer.histogram("train_hist", res_step, bins=5, step=step)
    return params
