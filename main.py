# TODO imports on top


def house_prices_train():
    # TODO load the train file
    # TODO load the embedding files
    # TODO call preprocessing utils on train
    # TODO call preprocessing utils on test using info
    # TODO save info from params
    # TODO dataloader this
    # TODO model + optimizer init
    # TODO training loop with logging and checkpoints
    # TODO print top K validation accs
    pass

def house_prices_test():
    # TODO load test file
    # TODO load embedding files and info
    # TODO preprocess data
    # TODO init model
    # TODO load desired checkpoint into model
    # TODO make predictions and save
    pass


if __name__ == "__main__":
    import optax
    import jax.random as random

    from data_loader import get_dataset, linear_data, train_test_split
    import models
    from train import train
    from utils import mse_loss

    n_features = 10
    n_layers = 3
    lr = 5 * 1e-3
    num_epochs = 500
    batch_size = 32
    single_batch = True
    n_datapoints = 200
    bias = 15.15
    data_seed = 123415
    train_seed = 123414651
    test_share = 0.3
    model_selected = "Resnet"

    X, Y = linear_data(
        seed=data_seed, n=n_datapoints, p=n_features, bias=bias,
    )
    full_dataset = get_dataset(
        X, Y, batch_size=batch_size, buffer_size=n_datapoints, numpy=False,
    )
    train_data, test_data = train_test_split(
        full_dataset, test_share=test_share
    )
    train(
        rng=random.PRNGKey(train_seed),
        model=getattr(models, model_selected)(
            [n_features for _ in range(n_layers)] + [1],
            dropout=False,
            dropout_rate=0.1,
        ),
        optimizer=optax.adam(lr),
        train_dataset=train_data,
        eval_dataset=test_data,
        loss_fn=mse_loss,
        num_epochs=num_epochs,
        inputs_shape=(n_features,),
        bias=bias,
        layer_name=str(n_layers),
        print_every=1,
        output_dir="logs",
        hist_every=1,
        single_batch=single_batch,
    )
else:
    import sys

    print("main.py should not be imported and used only as a script.")
    sys.exit(1)
