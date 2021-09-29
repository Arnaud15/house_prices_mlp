"""
Simple training module in Flax.
"""

import os

import jax
import jax.numpy as jnp
import jax.random as random
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training import checkpoints

from board import SummaryWriter
from models import init_params
from train_state import TrainStateWithLoss
from train_utils import update_running


@jax.jit
def train_step(rng, x_num, x_cat, y, state: TrainStateWithLoss):
    """Updates the training state on a batch (x_num, x_cat, y) of data.

    We can jit this function because our state subclasses a jax PyTreeNode.
    """

    def pure_loss(params):
        predicted = state.apply_fn(
            params, x_num, x_cat, rngs={"dropout": rng}
        )  # TODO generalize this dropout arg
        loss_items = state.loss_fn(y, predicted)
        return jnp.mean(loss_items), predicted

    (loss_step, predicted_step), grad_step = jax.value_and_grad(
        pure_loss, has_aux=True
    )(state.params)
    state = state.apply_gradients(grads=grad_step)
    residuals_step = y - predicted_step
    return state, loss_step, residuals_step


@jax.jit
def eval_step(rng, x_num, x_cat, y, state: TrainStateWithLoss):
    """Evaluates the current training state on a batch (x_num, x_cat, y) of data.

    We can jit this function because our state subclasses a jax PyTreeNode.
    """

    def pure_loss(params):
        predicted = state.apply_fn(
            params, x_num, x_cat, rngs={"dropout": rng}, train=False
        )  # TODO generalize this rngs arg
        loss_items = state.loss_fn(y, predicted)
        return jnp.mean(loss_items), predicted

    loss_step, predicted_step = pure_loss(state.params)
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
    num_input_shape,
    cat_input_shape,
    smoothing_alpha: float = 0.9,
    hist_every=None,
    print_every=None,
):
    if not os.path.isdir("logs"):
        os.makedirs("logs")
        print(f"Directory 'logs' did not exist and was created.")
    writer = SummaryWriter("logs")
    if hist_every is not None:
        assert isinstance(hist_every, int)
    if print_every is not None:
        assert isinstance(print_every, int)

    rng, init_params_rng = random.split(rng, num=2)
    params = init_params(
        init_params_rng,
        model,
        num_input_shape,
        cat_input_shape,
        dropout=model.dropout,
    )

    train_state = TrainStateWithLoss(
        step=0,
        apply_fn=model.apply,
        loss_fn=loss_fn,
        params=params,
        tx=optimizer,
        opt_state=optimizer.init(params),
    )

    step = 0
    running_train_loss = None
    for epoch_ix in range(num_epochs):
        for (x_num, x_cat, y) in train_dataset:
            rng, rng_step = random.split(rng, num=2)
            train_state, loss_step, res_step = train_step(
                rng=rng_step, x_num=x_num, x_cat=x_cat, y=y, state=train_state,
            )
            running_train_loss = update_running(
                obs=loss_step, loss=running_train_loss, decay=smoothing_alpha,
            )
            if (print_every is not None) and (step % print_every == 0):
                writer.scalar("train_loss", running_train_loss, step=step)
                print(f"Step {step} | Training Loss: {loss_step:.4f}")
            if (hist_every is not None) and (step % hist_every == 0):
                writer.histogram("train_hist", res_step, bins=5, step=step)

            step += 1
        eval_loss = 0.0
        eval_batches = 0
        for (x_num, x_cat, y) in eval_dataset:
            rng, rng_step = random.split(rng, num=2)
            loss_eval_step, _ = eval_step(
                rng_step, x_num=x_num, x_cat=x_cat, y=y, state=train_state,
            )
            eval_loss += loss_eval_step
            eval_batches += 1
        eval_loss /= eval_batches
        writer.scalar("validation_loss", eval_loss, step=epoch_ix)
        print(f"Epoch {epoch_ix + 1} | Validation Loss: {eval_loss:.4f}")
    return params
