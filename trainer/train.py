import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from stable_baselines3 import DQN

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
POSITION_OPENING_PENALTY = 0.01

STOCK_MAX_DAYS_NUM = None  # YEARLY_TRADABLE_DAYS_NUM * 20
STOCK_SYMBOLS = ["AAPL"]
ZIGZAG_PUBLISHED_DATE = datetime.datetime.strptime("2000-01-01", "%Y-%m-%d").date()


def generate_stock_assets() -> List[Stock]:
    assets = [
        Stock(
            symbol, Path(__file__).parent.parent / "data" / "stock" / "input" / "us",
            max_days_num=STOCK_MAX_DAYS_NUM,
        )
        for symbol in STOCK_SYMBOLS
    ]
    return assets


def generate_zigzag_assets(assets_num: int) -> List[Zigzag]:
    assets = [
        Zigzag(
            datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            ZIGZAG_PUBLISHED_DATE, np.random.uniform(0, 100),
            (0.55, 0.45), (2, 12), (0.0, 0.0075), (-0.01, 0.01),
        )
        for _ in range(assets_num)
    ]
    return assets


def generate_envs(
    train_asset_pool_generator: Callable[[], AssetPool],
    val_asset_pool_generator: Callable[[], AssetPool],
    test_asset_pool_generator: Callable[[], AssetPool],
) -> Tuple[TradingPlatform, Dict[str, TradingPlatform]]:
    # Training environment
    train_asset_pool = train_asset_pool_generator()
    train_env = TradingPlatform(
        train_asset_pool, HISTORICAL_DAYS_NUM,
        position_opening_penalty=POSITION_OPENING_PENALTY,
        max_balance_loss=1.0, max_balance_gain=0.5, max_positions_num=50, max_steps_num=YEARLY_TRADABLE_DAYS_NUM,
    )
    train_env.is_training = True
    train_env.apply_date_range((None, LAST_TRAINING_DATE))
    rep_train_env = TradingPlatform(
        train_asset_pool, HISTORICAL_DAYS_NUM,
        position_opening_penalty=POSITION_OPENING_PENALTY,
    )
    rep_train_env.is_training = False
    rep_train_env.apply_date_range((None, LAST_TRAINING_DATE))
    # Validation environment
    val_asset_pool = val_asset_pool_generator()
    val_env = TradingPlatform(
        val_asset_pool, HISTORICAL_DAYS_NUM,
        position_opening_penalty=POSITION_OPENING_PENALTY,
    )
    val_env.is_training = False
    val_env.apply_date_range((LAST_TRAINING_DATE, LAST_VALIDATION_DATE), excluding_historical=False)
    # Test environment
    test_asset_pool = test_asset_pool_generator()
    test_env = TradingPlatform(
        test_asset_pool, HISTORICAL_DAYS_NUM,
        position_opening_penalty=POSITION_OPENING_PENALTY,
    )
    test_env.is_training = False
    test_env.apply_date_range((LAST_VALIDATION_DATE, None), excluding_historical=False)
    return train_env, {"train": rep_train_env, "val": val_env, "test": test_env}


if __name__ == "__main__":
    train_env, eval_envs = generate_envs(
        lambda: AssetPool(
            generate_zigzag_assets(1),
            asset_generator=lambda: generate_zigzag_assets(1),
            polarity_temperature=5.0,
        ),
        lambda: AssetPool(generate_zigzag_assets(5), polarity_temperature=5.0),
        lambda: AssetPool(generate_zigzag_assets(5), polarity_temperature=5.0),
    )
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model = DQN(
        "MultiInputPolicy", train_env,
        gamma=0.95, policy_kwargs={"net_arch": [64, 64, 64]},
        verbose=1,
    )
    model.learn(
        total_timesteps=2000000,
        callback=FullEvalCallback(
            Path(__file__).parent.parent / "data" / "zigzag" / "output" / now,
            eval_envs, 100,
            showing_image=False
        ),
        log_interval=1000,
    )
