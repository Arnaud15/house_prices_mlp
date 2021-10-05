# Flax / Jax Practice with Housing Prices

Install dependencies locally, dev-install the package, download data (only needs to be done once).

Get the dataset [on Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

```
# from repo root
poetry install
# requires kaggle creds
mkdir -p data; cd data
poetry run kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip
cp transforms data/
```

Running tests
```
poetry run pytest
```

Interactive jupyter notebooks
```
poetry run jupyter notebook
```

Training
```
poetry run python -m house_prices_mlp.main
```
