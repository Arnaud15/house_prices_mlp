import json
import os

import numpy as np
import pandas as pd

from data_loader import get_dataset
from models import MLP
from preprocess_data import preprocess_eval, preprocess_train

EVAL_SHARE = 0.3

# TODOs
# embeddings
# training params
# logging and observability


def house_prices_train(training_params):
    train_data_full = pd.read_csv(os.path.join("data", "train.csv")).iloc[
        :, 1:
    ]  # remove first id column

    np.random.seed(151515)
    train_indices = np.random.rand(len(train_data_full)) >= EVAL_SHARE
    train_data, eval_data = (
        train_data_full[train_indices],
        train_data_full[~train_indices],
    )
    assert len(train_data) + len(eval_data) == len(train_data_full)

    transforms_info = None
    with open(os.path.join("data", "transforms"), "r") as f:
        transforms_info = json.loads(f)

    X_train, y_train, preproc_info = preprocess_train(
        train_data, transforms_info
    )
    with open(
        os.path.join("data", f"preprocess_info_{training_params.uid}"), "w"
    ) as f:
        json.dumps(f, preproc_info)
    X_eval, y_eval = preprocess_eval(eval_data, preproc_info)

    dataset_train = get_dataset(
        X_train,
        y_train,
        training_params.batch_size,
        len(train_data),
        numpy=False,
    )
    dataset_eval = get_dataset(
        X_eval, y_eval, training_params.batch_size, len(eval_data), numpy=False
    )
    model = (
        MLP(
            [
                training_params.layer_size
                for _ in range(training_params.n_layers)
            ]
            + [1],
            dropout=training_params.dropout,
            dropout_rate=training_params.dropout_rate,
            batch_norm=training_params.batch_norm,
        ),
    )
    train(
        rng=random.PRNGKey(train_seed),
        model=model,
        optimizer=optax.adam(training_params.lr),
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        loss_fn=mse_loss,
        num_epochs=training_params.num_epochs,
        inputs_shape=(X_train.shape[1],),
        bias=y_train.mean(),
        layer_name=str(training_params.n_layers),
        print_every=1,
        output_dir="logs",
        hist_every=1,
        single_batch=single_batch,
    )

    # TODO training loop with logging and checkpoints
    # TODO print top K validation accs
    pass


def house_prices_test(training_params):
    # TODO load test file
    # TODO load embedding files and info
    # TODO preprocess data
    # TODO init model
    # TODO load desired checkpoint into model
    # TODO make predictions and save
    pass


if __name__ == "__main__":
    print("coming soon")
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
