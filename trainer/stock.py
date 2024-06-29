import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from stable_baselines3 import DQN

from trainer.asset.stock import Stock
from trainer.env.asset_pool import AssetPool
from trainer.env.trading_platform import TradingPlatform
from trainer.env.evaluation import FullEvalCallback


MONTHLY_TRADABLE_DAYS_NUM = 20
YEARLY_TRADABLE_DAYS_NUM = 250
MAX_DAYS_NUM = None  # YEARLY_TRADABLE_DAYS_NUM * 20
LAST_TRAINING_DATE = datetime.datetime.strptime("2019-12-31", "%Y-%m-%d").date()
LAST_VALIDATION_DATE = datetime.datetime.strptime("2022-12-31", "%Y-%m-%d").date()
HISTORICAL_DAYS_NUM = MONTHLY_TRADABLE_DAYS_NUM * 6
POSITION_OPENING_FEE = 0.01
SYMBOLS = ["AAPL"]


def generate_asset_pool(symbols: List[str]) -> AssetPool:
    assets = [Stock(symbol, Path("../data/input/stock/us"), max_days_num=MAX_DAYS_NUM) for symbol in symbols]
    return AssetPool(assets, polarity_temperature=5.0)


def generate_envs() -> Tuple[TradingPlatform, Dict[str, TradingPlatform]]:
    # Training environment
    train_env = TradingPlatform(
        generate_asset_pool(SYMBOLS), HISTORICAL_DAYS_NUM,
        position_opening_fee=POSITION_OPENING_FEE,
        max_balance_loss=1.0, max_balance_gain=0.5, max_positions_num=50, max_steps_num=YEARLY_TRADABLE_DAYS_NUM,
    )
    train_env.is_training = True
    train_env.apply_date_range(max_date=LAST_TRAINING_DATE)
    rep_train_env = TradingPlatform(
        generate_asset_pool(SYMBOLS), HISTORICAL_DAYS_NUM,
        position_opening_fee=POSITION_OPENING_FEE,
    )
    rep_train_env.is_training = False
    rep_train_env.apply_date_range(max_date=LAST_TRAINING_DATE)
    # Validation environment
    val_env = TradingPlatform(
        generate_asset_pool(SYMBOLS), HISTORICAL_DAYS_NUM,
        position_opening_fee=POSITION_OPENING_FEE,
    )
    val_env.is_training = False
    val_env.apply_date_range(min_date=LAST_TRAINING_DATE, max_date=LAST_VALIDATION_DATE, excluding_historical=False)
    # Test environment
    test_env = TradingPlatform(
        generate_asset_pool(SYMBOLS), HISTORICAL_DAYS_NUM,
        position_opening_fee=POSITION_OPENING_FEE,
    )
    test_env.is_training = False
    test_env.apply_date_range(min_date=LAST_VALIDATION_DATE, excluding_historical=False)
    return train_env, {"train": rep_train_env, "val": val_env, "test": test_env}


# Start training
train_env, eval_envs = generate_envs()
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model = DQN(
    "MultiInputPolicy", train_env,
    gamma=0.95, policy_kwargs={"net_arch": [64, 64, 64]},
    verbose=1,
)
model.learn(
    total_timesteps=2000000,
    callback=FullEvalCallback(Path(f"../data/output/{now}"), eval_envs, 100, showing_image=False),
    log_interval=1000,
)
