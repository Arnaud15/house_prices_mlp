from collections import namedtuple
import json
import os

import numpy as np
import pandas as pd

from preprocess import get_transformed_data
from data_loader import get_dataset, train_test_split_pandas
from models import CustomMLP
from training_loop import train
from train_utils import mse_loss

import jax.random as random
import optax


EVAL_SHARE = 0.3

# TODOs
# clean up the pipeline
# allow the model to embed stuff
# clean up logging with clu
# debug single training
# re-org files
# submission script
# multiple model checkpoints and logging handling
# debug a couple of regularization ideas
# ray notebook or script
# submissions with better and better models


def house_prices_train(args):
    seed = 1515151555151551551
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
        x_cat=train_transformed.X_cat,
        y_data=train_transformed.y,
        batch_size=args.batch_size,
        buffer_size=len(train_data),
        numpy=True,
    )

    eval_dataset = get_dataset(
        x_num=eval_transformed.X_num,
        x_cat=eval_transformed.X_cat,
        y_data=eval_transformed.y,
        batch_size=args.batch_size,
        buffer_size=len(eval_data),
        numpy=True,
    )
    return train(
        rng=random.PRNGKey(seed),
        optimizer=optax.adam(args.lr),
        model=CustomMLP(
            layer_sizes=[args.hidden_size for _ in range(args.n_hidden_layers)],
            embedding_sizes=[(card + 1, args.embed_size) for card in cardinalities],
            dropout_rate=args.dropout_rate,
            dropout=args.dropout_enabled,
            bias=train_transformed.y.mean(),
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_fn=mse_loss,
        num_epochs=args.num_epochs,
        inputs_shape=(
            train_transformed.X_num.shape[1]
            + train_transformed.X_cat.shape[1] * args.embed_size,
        ),
        print_every=1,
        hist_every=1,
        single_batch=args.single_batch,
    )


if __name__ == "__main__":
    print("coming soon")
    Args = namedtuple(
        "Args",
        [
            "embed_size",
            "batch_size",
            "lr",
            "n_layers",
            "hidden_size",
            "dropout_enabled",
            "dropout_rate",
            "single_batch",
        ],
    )
    # n_features = 10
    # n_layers = 3
    # lr = 5 * 1e-3
    # num_epochs = 500
    # batch_size = 32
    # single_batch = True
    # n_datapoints = 200
    # bias = 15.15
    # data_seed = 123415
    # train_seed = 123414651
    # test_share = 0.3
    # model_selected = "Resnet"

    # X, Y = linear_data(
    #     seed=data_seed, n=n_datapoints, p=n_features, bias=bias,
    # )
    # full_dataset = get_dataset(
    #     X, Y, batch_size=batch_size, buffer_size=n_datapoints, numpy=False,
    # )
    # train_data, test_data = train_test_split(
    #     full_dataset, test_share=test_share
    # )
    # train(
    #     rng=random.PRNGKey(train_seed),
    #     model=getattr(models, model_selected)(
    #         [n_features for _ in range(n_layers)] + [1],
    #         dropout=False,
    #         dropout_rate=0.1,
    #     ),
    #     optimizer=optax.adam(lr),
    #     train_dataset=train_data,
    #     eval_dataset=test_data,
    #     loss_fn=mse_loss,
    #     num_epochs=num_epochs,
    #     inputs_shape=(n_features,),
    #     bias=bias,
    #     layer_name=str(n_layers),
    #     print_every=1,
    #     output_dir="logs",
    #     hist_every=1,
    #     single_batch=single_batch,
    # )
else:
    import sys

    print("main.py should not be imported and used only as a script.")
    sys.exit(1)
