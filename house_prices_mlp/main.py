import json
import os
import sys
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import lightgbm as lgb
import numpy as np
import optax
import pandas as pd

from .data_loader import get_dataset, train_test_split_pandas
from .hp_tuning import Args, GBDTArgs
from .models import CustomMLP
from .preprocess import Datapoints, get_transformed_data
from .training_loop import train

Score = float


def house_prices_data() -> Tuple[
    pd.DataFrame, Datapoints, Datapoints, Datapoints, List[int]
]:
    """Read raw data from csv, create a validation test (seeded), transform the data."""
    seed = 1515151555
    train_data_full = pd.read_csv(os.path.join("data", "train.csv")).iloc[
        :, 1:
    ]  # remove unused id column for training
    test_data_full = pd.read_csv(
        os.path.join("data", "test.csv")
    )  # keep first id column for the later submission

    # Load transforms
    transforms = None
    with open(os.path.join("data", "transforms"), "r") as f:
        transforms = json.load(f)
        transforms = [tuple(list_) for list_ in transforms]

    # Seed the train / val split
    np.random.seed(seed)
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
    return (
        test_data_full,
        train_transformed,
        eval_transformed,
        test_transformed,
        cardinalities,
    )


def submit_predictions(
    sub_name: str, predictions: jnp.ndarray, id_col: jnp.array
):
    """Helper function to write predictions to a file in a format compatible
    with Kaggle's requirements."""
    with open(os.path.join("data", sub_name), "w") as sub_file:
        sub_file.write("Id,SalePrice\n")
        for (example_id, pred) in zip(id_col, jnp.squeeze(predictions)):
            sub_file.write(f"{example_id},{pred}\n")


def to_design(transformed_data) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Helper function to get a design matrix and target vector.
    The target vector is `None` for test data.
    """
    X = np.concatenate(
        [transformed_data.X_cat, transformed_data.X_num], axis=1
    )
    y = transformed_data.y
    if y is not None:
        assert X.shape[0] == len(transformed_data.y)
    return X, y


def house_prices_gbdt(gbdt_args: GBDTArgs) -> Tuple[Score, GBDTArgs]:
    """End-to-end data preprocessing, training and testing script for Gradient
    Boosting with Decision Trees model."""
    (
        test_data_full,
        train_transformed,
        eval_transformed,
        test_transformed,
        _,
    ) = house_prices_data()
    lgb_regressor = lgb.LGBMRegressor(
        boosting_type="gbdt",
        num_leaves=32
        if gbdt_args.max_depth == -1
        else gbdt_args.max_depth ** 2,
        max_depth=gbdt_args.max_depth,
        learning_rate=gbdt_args.learning_rate,
        n_estimators=gbdt_args.num_rounds,
        subsample=gbdt_args.bagging_fraction,
        reg_alpha=gbdt_args.lambda_l1,
    )
    n_cat = train_transformed.X_cat.shape[1]
    X_train, y_train = to_design(train_transformed)
    X_eval, y_eval = to_design(eval_transformed)
    lgb_regressor.fit(
        X_train,
        y_train,
        eval_set=[(X_eval, y_eval)],
        eval_metric="rmse",
        early_stopping_rounds=gbdt_args.early_stopping_round,
        categorical_feature=list(range(n_cat)),
    )
    best_score = lgb_regressor.best_score_["valid_0"]["rmse"]
    print(
        f"Training completed, best evaluation performance is {best_score:.3f}"
    )
    X_test, _ = to_design(test_transformed)
    assert X_test.shape[0] == len(test_data_full)
    predictions = np.exp(lgb_regressor.predict(X_test,))
    print(
        f"predictions mean: {predictions.mean():.3f}, std: {predictions.std():.3f}, min: {predictions.min():.3f}, max: {predictions.max():.3f}."
    )
    sub_name = f"lgb_submission_{hash(gbdt_args)}"
    submit_predictions(
        sub_name,
        predictions=predictions,
        id_col=test_data_full.loc[:, "Id"].values,
    )
    return best_score, sub_name


def house_prices_mlp(args: Args) -> Tuple[Score, Args]:
    """End-to-end data preprocessing, training and testing script for
    regularized MLP model."""
    (
        test_data_full,
        train_transformed,
        eval_transformed,
        test_transformed,
        cardinalities,
    ) = house_prices_data()
    train_dataset = get_dataset(
        x_num=train_transformed.X_num,
        x_cat=train_transformed.X_cat.astype(int),
        y_data=train_transformed.y,
        batch_size=args.batch_size,
        buffer_size=len(train_transformed.y),
        single_batch=args.single_batch,
    )
    eval_dataset = get_dataset(
        x_num=eval_transformed.X_num,
        x_cat=eval_transformed.X_cat.astype(int),
        y_data=eval_transformed.y,
        batch_size=len(eval_transformed.y),
        buffer_size=len(eval_transformed.y),
        single_batch=False,
    )
    model = CustomMLP(
        layer_sizes=[args.hidden_size for _ in range(args.n_layers)] + [1],
        vocab_sizes=[card + 1 for card in cardinalities],
        embed_size=args.embed_size,
        dropout_rate=args.dropout_rate,
        dropout=args.dropout_enabled,
        bias=train_transformed.y.mean(),
        batch_norm=args.batch_norm,
        residuals=args.resnet,
    )
    trained_params, eval_loss = train(
        rng=random.PRNGKey(12345),
        model=model,
        optimizer=optax.adamw(
            args.lr,
            weight_decay=args.decay_rate,
            mask=lambda params: jax.tree_map(lambda x: x.ndim > 1, params),
        )
        if args.weight_decay
        else optax.adam(args.lr),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=args.n_epochs,
        cat_input_shape=(args.batch_size, train_transformed.X_cat.shape[1],),
        num_input_shape=(args.batch_size, train_transformed.X_num.shape[1],),
        hist_every=1,
        print_every=args.print_every,
    )
    print(f"Evaluation RMSE after training: {eval_loss:.3f}")
    rng = random.PRNGKey(1513241)
    predictions = jnp.exp(
        model.apply(
            trained_params,
            test_transformed.X_num,
            test_transformed.X_cat.astype(int),
            rngs={"dropout": rng},
            train=False,
        )
    )
    print(
        f"predictions mean: {predictions.mean():.3f}, std: {predictions.std():.3f}, min: {predictions.min():.3f}, max: {predictions.max():.3f}."
    )
    sub_name = f"submission_{hash(args)}"
    submit_predictions(
        sub_name,
        predictions=predictions,
        id_col=test_data_full.loc[:, "Id"].values,
    )
    return eval_loss, sub_name


if __name__ == "__main__":
    from argparse import ArgumentParser

    import ray

    ray.init()
    from .hp_tuning import random_gbdt_params, random_params

    def hp_search_ray_gbdt(n_repeats: int) -> Tuple[np.ndarray, str]:
        @ray.remote
        def closure():
            args = random_gbdt_params()
            return house_prices_gbdt(args)

        out = ray.get([closure.remote() for _ in range(n_repeats)])
        return sorted(out, key=lambda x: x[0])

    def hp_search_ray_mlp(n_repeats: int) -> Tuple[np.ndarray, str]:
        @ray.remote
        def closure():
            args = random_params()
            return house_prices_mlp(args)

        out = ray.get([closure.remote() for _ in range(n_repeats)])
        return sorted(out, key=lambda x: x[0])[0]

    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specify a model, choice of 'mlp' or 'gbdt'",
        choices=["mlp", "gbdt"],
    )
    parser.add_argument(
        "--nrepeats",
        type=int,
        required=True,
        help="Specify a number of random retries for the hyperparameters search.",
    )
    args = parser.parse_args()

    assert (
        int(args.nrepeats) > 0
    ), f"number of repeats cannot be interpreted as a positive integer"
    if args.model == "mlp":
        print("Starting hyperparameters search for regularized MLPs.")
        hp_search_ray_mlp(args.nrepeats)
    else:
        assert args.model == "gbdt"
        print("Starting hyperparameters search for GBDTs.")
        hp_search_ray_gbdt(args.nrepeats)
else:
    print("main.py should not be imported and used only as a script.")
    sys.exit(1)
