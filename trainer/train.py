import datetime
import string
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from stable_baselines3 import DQN

from trainer.asset.base import DailyAsset
from trainer.asset.stock import Stock
from trainer.asset.zigzag import Zigzag
from trainer.env.asset_pool import AssetPool
from trainer.env.trading_platform import TradingPlatform
from trainer.env.evaluation import FullEvalCallback


MONTHLY_TRADABLE_DAYS_NUM = 20
YEARLY_TRADABLE_DAYS_NUM = 250
LAST_TRAINING_DATE = datetime.datetime.strptime("2019-12-31", "%Y-%m-%d").date()
LAST_VALIDATION_DATE = datetime.datetime.strptime("2022-12-31", "%Y-%m-%d").date()
HISTORICAL_DAYS_NUM = MONTHLY_TRADABLE_DAYS_NUM * 6

# LINK: Ignore the last position type (SIDELINE), use only BUY and SELL
INTEREST_RATE = 0.0  # Each year
POSITION_HOLDING_DAILY_FEE = INTEREST_RATE / YEARLY_TRADABLE_DAYS_NUM
POSITION_OPENING_PENALTY = 0.01

STOCK_MAX_DAYS_NUM = None  # YEARLY_TRADABLE_DAYS_NUM * 20
STOCK_SYMBOLS = ["AAPL"]
ZIGZAG_PUBLISHED_DATE = datetime.datetime.strptime("2000-01-01", "%Y-%m-%d").date()


def generate_envs(assets: List[DailyAsset]) -> Tuple[TradingPlatform, Dict[str, TradingPlatform]]:
    # Training environment
    train_asset_pool = AssetPool(
        assets,
        secondary_asset_generator=lambda: generate_zigzag_assets(1),
        # NOTE: This param is nearly meaningless if `ahead_days_num` param of `apply_date_range` method below
        # differs significantly from `train_env._max_steps_num`
        polarity_temperature=5.0,
    )
    train_asset_pool.apply_date_range(
        (None, LAST_TRAINING_DATE), HISTORICAL_DAYS_NUM,
        ahead_days_num=YEARLY_TRADABLE_DAYS_NUM,  # TODO: Choose the same value as `train_env._max_steps_num`
    )
    train_env = TradingPlatform(
        train_asset_pool, HISTORICAL_DAYS_NUM,
        position_holding_daily_fee=POSITION_HOLDING_DAILY_FEE, position_opening_penalty=POSITION_OPENING_PENALTY,
        max_balance_loss=1.0,
    )
    train_env.is_training = True
    rep_train_env = TradingPlatform(
        train_asset_pool, HISTORICAL_DAYS_NUM,
        position_holding_daily_fee=POSITION_HOLDING_DAILY_FEE, position_opening_penalty=POSITION_OPENING_PENALTY,
    )
    rep_train_env.is_training = False
    # Validation environment
    val_asset_pool = AssetPool(assets)
    val_asset_pool.apply_date_range(
        (LAST_TRAINING_DATE, LAST_VALIDATION_DATE), HISTORICAL_DAYS_NUM,
        excluding_historical=False,
    )
    val_env = TradingPlatform(
        val_asset_pool, HISTORICAL_DAYS_NUM,
        position_holding_daily_fee=POSITION_HOLDING_DAILY_FEE, position_opening_penalty=POSITION_OPENING_PENALTY,
    )
    val_env.is_training = False
    # Test environment
    test_asset_pool = AssetPool(assets)
    test_asset_pool.apply_date_range(
        (LAST_VALIDATION_DATE, None), HISTORICAL_DAYS_NUM,
        excluding_historical=False,
    )
    test_env = TradingPlatform(
        test_asset_pool, HISTORICAL_DAYS_NUM,
        position_holding_daily_fee=POSITION_HOLDING_DAILY_FEE, position_opening_penalty=POSITION_OPENING_PENALTY,
    )
    test_env.is_training = False
    return train_env, {"train": rep_train_env, "val": val_env, "test": test_env}


def generate_stock_assets(stock_symbols: List[str]) -> List[Stock]:
    assets = [
        Stock(
            symbol, Path(__file__).parent.parent / "data" / "stock" / "input" / "us",
            max_days_num=STOCK_MAX_DAYS_NUM,
        )
        for symbol in stock_symbols
    ]
    return assets


def generate_zigzag_assets(assets_num: int) -> List[Zigzag]:
    assets = [
        Zigzag(
            datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_"
            + "".join(np.random.choice([*(string.ascii_letters + string.digits)], size=4)),
            ZIGZAG_PUBLISHED_DATE, np.random.uniform(0, 10),
            (0.55, 0.45), (2, 6), (0.0025, 0.005), (-0.02, 0.02),
        )
        for _ in range(assets_num)
    ]
    return assets


def train(env_type: str):
    train_env: TradingPlatform
    eval_envs: Dict[str, TradingPlatform]
    if env_type == "stock":
        train_env, eval_envs = generate_envs(generate_stock_assets(STOCK_SYMBOLS))
    elif env_type == "zigzag":
        train_env, eval_envs = generate_envs(generate_zigzag_assets(5))
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model = DQN(
        "MultiInputPolicy", train_env,
        gamma=0.95, policy_kwargs={"net_arch": [64, 64, 64]},
        exploration_fraction=0.02,
        verbose=1,
    )
    model.learn(
        total_timesteps=20000000,
        callback=FullEvalCallback(
            Path(__file__).parent.parent / "data" / env_type / "output" / now,
            eval_envs, 100,
            showing_image=False,
        ),
        log_interval=100,
    )
