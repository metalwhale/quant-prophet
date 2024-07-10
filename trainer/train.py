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


def generate_envs(assets: List[DailyAsset]) -> Tuple[TradingPlatform, Dict[str, TradingPlatform]]:
    MONTHLY_TRADABLE_DAYS_NUM = 20
    YEARLY_TRADABLE_DAYS_NUM = 250
    LAST_TRAINING_DATE = datetime.datetime.strptime("1999-12-31", "%Y-%m-%d").date()
    EVAL_DATE_RANGES = [
        datetime.datetime.strptime(d, "%Y-%m-%d").date() if d is not None else None
        for d in ["2004-12-31", "2009-12-31", "2014-12-31", "2019-12-31", None]
    ]
    HISTORICAL_DAYS_NUM = MONTHLY_TRADABLE_DAYS_NUM * 6
    YEARLY_INTEREST_RATE = 0.0  # LINK: Ignore the last position type (SIDELINE), use only BUY and SELL
    POSITION_HOLDING_DAILY_FEE = YEARLY_INTEREST_RATE / YEARLY_TRADABLE_DAYS_NUM
    POSITION_OPENING_PENALTY = 0.01
    # Training environment
    train_asset_pool = AssetPool(
        assets,
        secondary_asset_generator=lambda: generate_zigzag_assets(1),
    )
    train_asset_pool.apply_date_range(
        (None, LAST_TRAINING_DATE), HISTORICAL_DAYS_NUM,
        ahead_days_num=YEARLY_TRADABLE_DAYS_NUM,  # TODO: Choose the same value as `train_env._max_steps_num`
    )
    train_env = TradingPlatform(
        train_asset_pool, HISTORICAL_DAYS_NUM,
        position_holding_daily_fee=POSITION_HOLDING_DAILY_FEE, position_opening_penalty=POSITION_OPENING_PENALTY,
    )
    train_env.is_training = True
    rep_train_env = TradingPlatform(
        train_asset_pool, HISTORICAL_DAYS_NUM,
        position_holding_daily_fee=POSITION_HOLDING_DAILY_FEE, position_opening_penalty=POSITION_OPENING_PENALTY,
    )
    rep_train_env.is_training = False
    # Evaluation environments
    eval_envs: Dict[str, TradingPlatform] = {"0train": rep_train_env}  # Use train env for evaluation as well
    for i, last_eval_date in enumerate(EVAL_DATE_RANGES):
        eval_asset_pool = AssetPool(assets)
        eval_asset_pool.apply_date_range(
            (LAST_TRAINING_DATE if i == 0 else EVAL_DATE_RANGES[i - 1], last_eval_date), HISTORICAL_DAYS_NUM,
            excluding_historical=False,
        )
        eval_env = TradingPlatform(
            eval_asset_pool, HISTORICAL_DAYS_NUM,
            position_holding_daily_fee=POSITION_HOLDING_DAILY_FEE, position_opening_penalty=POSITION_OPENING_PENALTY,
        )
        eval_env.is_training = False
        eval_envs[f"{i + 1}eval"] = eval_env
    return train_env, eval_envs


def generate_stock_assets() -> List[Stock]:
    MAX_DAYS_NUM = None  # YEARLY_TRADABLE_DAYS_NUM * 20
    SYMBOLS: List[str] = ["AAPL"]
    assets = [
        Stock(
            symbol, Path(__file__).parent.parent / "data" / "stock" / "input" / "us",
            max_days_num=MAX_DAYS_NUM,
        )
        for symbol in SYMBOLS
    ]
    return assets


def generate_zigzag_assets(assets_num: int) -> List[Zigzag]:
    PUBLISHED_DATE = datetime.datetime.strptime("1980-01-01", "%Y-%m-%d").date()  # Before `LAST_TRAINING_DATE`
    assets = [
        Zigzag(
            f"zigzag_{i}" + "".join(np.random.choice([*(string.ascii_letters + string.digits)], size=4)),
            PUBLISHED_DATE, np.random.uniform(0, 10),
            (0.55, 0.45), (2, 6), (0.005, 0.005), (-0.02, 0.02),
        )
        for i in range(assets_num)
    ]
    return assets


def train(env_type: str):
    train_env: TradingPlatform
    eval_envs: Dict[str, TradingPlatform]
    if env_type == "stock":
        train_env, eval_envs = generate_envs(generate_stock_assets() + generate_zigzag_assets(5))
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
