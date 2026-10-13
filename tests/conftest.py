import pytest

from churnprop.config import load_config
from churnprop.data import load_clean, make_split


@pytest.fixture(scope="session")
def cfg():
    """Fast config: 3 folds, 1 repeat, 100 trees, 10 epochs."""
    return load_config(fast=True, seed=7)


@pytest.fixture(scope="session")
def df(cfg):
    return load_clean(cfg)


@pytest.fixture(scope="session")
def split(cfg, df):
    return make_split(df, cfg)


@pytest.fixture(scope="session")
def small_split(cfg, df):
    """A 1,500-customer split so model tests run in seconds."""
    sub = df.sample(1500, random_state=0).reset_index(drop=True)
    return make_split(sub, cfg)
