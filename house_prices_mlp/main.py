import json
import os
import sys

import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
import pandas as pd

from data_loader import get_dataset, train_test_split_pandas
from hp_tuning import random_params
from models import CustomMLP
from preprocess import get_transformed_data
from training_loop import train

# TODOs
# script for rd hp search (ray)
# other karpathy checks
# debug a couple of regularization ideas
# submissions with better and better models
# re-org files
# if time: clu logging and ml collections params, lifted module


def house_prices_train(args):
    seed = 1515151555
    train_data_full = pd.read_csv(os.path.join("data", "train.csv")).iloc[
        :, 1:
    ]  # remove unused id column
    test_data_full = pd.read_csv(
        os.path.join("data", "test.csv")
    )  # keep first id column

    np.random.seed(seed)

    transforms = None
    with open(os.path.join("data", "transforms"), "r") as f:
        transforms = json.load(f)
        transforms = [tuple(list_) for list_ in transforms]

    train_data, eval_data = train_test_split_pandas(train_data_full)

    print(
        f"""Dataframe shapes (rows, cols):
train {len(train_data), len(train_data.columns)},
eval {len(eval_data), len(eval_data.columns)},
test {len(test_data_full), len(test_data_full.columns)}."""
    )

    (
        train_transformed,
        eval_transformed,
        test_transformed,
        cardinalities,
    ) = get_transformed_data(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data_full,
        transforms=transforms,
    )

    print(f"Baseline RMSE: {train_transformed.y.std():.3f}.")

    train_dataset = get_dataset(
        x_num=train_transformed.X_num,
        x_cat=train_transformed.X_cat.astype(int),
        y_data=train_transformed.y,
        batch_size=args.batch_size,
        buffer_size=len(train_data),
        single_batch=args.single_batch,
    )

    eval_dataset = get_dataset(
        x_num=eval_transformed.X_num,
        x_cat=eval_transformed.X_cat.astype(int),
        y_data=eval_transformed.y,
        batch_size=len(eval_data),
        buffer_size=len(eval_data),
        single_batch=False,
    )

    model = CustomMLP(
        layer_sizes=[args.hidden_size for _ in range(args.n_layers)] + [1],
        vocab_sizes=[card + 1 for card in cardinalities],
        embed_size=args.embed_size,
        dropout_rate=args.dropout_rate,
        dropout=args.dropout_enabled,
        bias=train_transformed.y.mean(),
    )

    trained_params, eval_loss = train(
        rng=random.PRNGKey(seed),
        model=model,
        optimizer=optax.adam(args.lr),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=args.n_epochs,
        cat_input_shape=(args.batch_size, train_transformed.X_cat.shape[1],),
        num_input_shape=(args.batch_size, train_transformed.X_num.shape[1],),
        hist_every=1,
        print_every=1,
    )
    print(f"Evaluation RMSE after training: {eval_loss:.3f}")

    rng = random.PRNGKey(1513241)
    predictions = jnp.exp(
        model.apply(
            trained_params,
            test_transformed.X_num,
            test_transformed.X_cat.astype(int),
            rngs={"dropout": rng},
        )
    )
    print(
        f"predictions mean: {predictions.mean():.3f}, std: {predictions.std():.3f}, min: {predictions.min():.3f}, max: {predictions.max():.3f}."
    )

    u_sub_name = f"submission_{hash(args)}"
    with open(os.path.join("data", u_sub_name), "w") as sub_file:
        sub_file.write("Id,SalePrice\n")
        for (example_id, pred) in zip(
            test_data_full.loc[:, "Id"].values, jnp.squeeze(predictions)
        ):
            sub_file.write(f"{example_id},{pred}\n")
    return eval_loss, u_sub_name


if __name__ == "__main__":
    import ray

    ray.init()
    try:
        n_repeats = int(sys.argv[1])
    except ValueError:
        print("n repeats cannot be interpreted as an integer")
        n_repeats = 1000
    except IndexError:
        print("n repeats not provided")
        n_repeats = 1000

    @ray.remote
    def closure():
        args = random_params()
        return house_prices_train(args), args

    out = ray.get([closure.remote() for _ in range(n_repeats)])
    out = sorted(out, key=lambda x: x[0][0])[0]
    print(out)
else:
    print("main.py should not be imported and used only as a script.")
    sys.exit(1)