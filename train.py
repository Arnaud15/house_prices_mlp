import jax
import jax.random as random
import jax.numpy as jnp
import optax

from utils import weight_decay


def train_step(
    params, opt_state, x, y, model, optimizer, loss_fn, metric_fn, l2_scale
):
    @jax.jit
    def pure_loss(params):
        predicted = model.apply(params, x)
        return jnp.mean(jnp.mean(loss_fn(y, predicted), 0)) + weight_decay(
            scale=l2_scale, params_tree=params
        )  # TODO no good reason to compute weight decay by default

    loss_step, grad_step = jax.value_and_grad(pure_loss)(params)
    param_updates, opt_state = optimizer.update(grad_step, opt_state)
    metric_step = jnp.mean(jnp.mean(metric_fn(y, model.apply(params, x)), 0))
    params = optax.apply_updates(params, param_updates)

    return params, opt_state, loss_step, metric_step


def train(
    rng,
    model,
    optimizer,
    dataset,
    loss_fn,
    metric_fn,
    num_epochs,
    inputs_shape,
    l2_scale=0.0,
):

    rng, init_key = random.split(rng)
    dummy_input = random.normal(init_key, shape=inputs_shape)

    rng, init_key = random.split(rng)
    params = model.init(init_key, dummy_input)

    opt_state = optimizer.init(params)

    for epoch_ix in range(num_epochs):
        step = 0
        for (x_train, y_train) in dataset:
            params, opt_state, loss_step, metric_step = train_step(
                params,
                opt_state,
                x_train,
                y_train,
                model,
                optimizer,
                loss_fn,
                metric_fn,
                l2_scale,
            )
            step += 1
            if step % 25 == 0:
                print(
                    f"Step {step}, Metric: {metric_step:.3f}, Loss: {loss_step:.3f}"
                )
    return params
