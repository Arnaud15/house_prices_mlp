import json
import os
from collections import namedtuple

import jax.random as random
import numpy as np
import optax
import pandas as pd

from data_loader import get_dataset, train_test_split_pandas
from models import CustomMLP
from preprocess import get_transformed_data
from train_utils import mse_loss
from training_loop import train

# TODOs
# script for rd hp search
# submission script
# debug a couple of regularization ideas
# ray notebook or script
# submissions with better and better models
# re-org files


def house_prices_train(args):
    seed = 1515151555
    train_data_full = pd.read_csv(os.path.join("data", "train.csv")).iloc[
        :, 1:
    ]  # remove first id column

    np.random.seed(seed)

    transforms = None
    with open(os.path.join("data", "transforms"), "r") as f:
        transforms = json.load(f)
        transforms = [tuple(list_) for list_ in transforms]

    train_data, eval_data = train_test_split_pandas(train_data_full)

    train_transformed, eval_transformed, cardinalities = get_transformed_data(
        train_data=train_data, eval_data=eval_data, transforms=transforms
    )

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

    trained_params = train(
        rng=random.PRNGKey(seed),
        model=model,
        optimizer=optax.adam(args.lr),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_fn=mse_loss,
        num_epochs=args.n_epochs,
        cat_input_shape=(args.batch_size, train_transformed.X_cat.shape[1],),
        num_input_shape=(args.batch_size, train_transformed.X_num.shape[1],),
        hist_every=1,
        print_every=1,
    )

    return model, trained_params


if __name__ == "__main__":
    Args = namedtuple(
        "Args",
        [
            "embed_size",
            "batch_size",
            "lr",
            "n_layers",
            "n_epochs",
            "hidden_size",
            "dropout_enabled",
            "dropout_rate",
            "single_batch",
        ],
    )
    args = Args(
        embed_size=3,
        batch_size=32,
        lr=1e-4,
        n_epochs=1000,
        n_layers=2,
        hidden_size=16,
        dropout_enabled=False,
        dropout_rate=0.1,
        single_batch=True,
    )
    params = house_prices_train(args)
else:
    import sys

    print("main.py should not be imported and used only as a script.")
    sys.exit(1)
