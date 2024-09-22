import datetime
import string
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from stable_baselines3 import DQN

from trainer.asset.base import DailyAsset
from trainer.asset.stock import Stock, StockIndex
from trainer.asset.zigzag import Zigzag
from trainer.env.asset_pool import AssetPool
from trainer.env.trading_platform import MONTHLY_TRADABLE_DAYS_NUM, TradingPlatform
from trainer.env.evaluation import FullEvalCallback


def generate_envs(assets: List[DailyAsset]) -> Tuple[TradingPlatform, Dict[str, TradingPlatform]]:
    FIRST_TRAINING_DATE = datetime.datetime.strptime("1990-01-01", "%Y-%m-%d").date()
    LAST_TRAINING_DATE = datetime.datetime.strptime("2009-12-31", "%Y-%m-%d").date()
    EVAL_DATE_RANGES = [
        datetime.datetime.strptime(d, "%Y-%m-%d").date() if d is not None else None
        for d in ["2014-12-31", "2019-12-31", None]
    ]
    HISTORICAL_DAYS_NUM = MONTHLY_TRADABLE_DAYS_NUM * 6
    assets = [a for a in assets if a.published_date <= FIRST_TRAINING_DATE]
    assets.sort(key=lambda asset: asset.symbol)
    # Training environment
    train_asset_pool = AssetPool(assets)
    train_asset_pool.apply_date_range((FIRST_TRAINING_DATE, LAST_TRAINING_DATE), HISTORICAL_DAYS_NUM)
    train_env = TradingPlatform(train_asset_pool, HISTORICAL_DAYS_NUM)
    train_env.set_mode(True)
    train_env.figure_num = "train"
    rep_train_env = TradingPlatform(train_asset_pool, HISTORICAL_DAYS_NUM)
    rep_train_env.set_mode(False)
    rep_train_env.figure_num = "train"
    # Evaluation environments
    eval_envs: Dict[str, TradingPlatform] = {"0train": rep_train_env}  # Use train env for evaluation as well
    for i, last_eval_date in enumerate(EVAL_DATE_RANGES):
        eval_asset_pool = AssetPool(assets)
        eval_asset_pool.apply_date_range(
            (LAST_TRAINING_DATE if i == 0 else EVAL_DATE_RANGES[i - 1], last_eval_date), HISTORICAL_DAYS_NUM,
            excluding_historical=False,
        )
        eval_env = TradingPlatform(eval_asset_pool, HISTORICAL_DAYS_NUM)
        eval_env.set_mode(False)
        eval_env.figure_num = "eval"  # All eval envs have the same `figure_num` to avoid creating too many figures
        eval_envs[f"{i + 1}eval"] = eval_env
    return train_env, eval_envs


def generate_stock_assets() -> List[Stock]:
    MAX_DAYS_NUM = None
    constituents_file_path = Path(__file__).parent.parent / "data" / "stock" / "input" / "sp500.csv"
    input_dir_path = Path(__file__).parent.parent / "data" / "stock" / "input" / "us"
    symbols = StockIndex(constituents_file_path, input_dir_path).symbols
    assets = [Stock(s, input_dir_path, max_days_num=MAX_DAYS_NUM) for s in symbols]
    return assets


def generate_zigzag_assets(published_date_str: str, assets_num: int) -> List[Zigzag]:
    PUBLISHED_DATE = datetime.datetime.strptime(published_date_str, "%Y-%m-%d").date()  # Before `LAST_TRAINING_DATE`
    assets = []
    for i in range(assets_num):
        up_weight = np.random.uniform(0.51, 0.53)
        trend_movement_scale = np.random.uniform(0.018, 0.022)
        assets.append(Zigzag(
            f"zigzag_{i}" + "".join(np.random.choice([*(string.ascii_letters + string.digits)], size=4)),
            PUBLISHED_DATE, np.random.uniform(0, 1),
            (up_weight, 1.0 - up_weight), (1, 2), (0, trend_movement_scale),
            (-0.0001, 0.0001),  # Small values are fine
        ))
    return assets


def train(env_type: str):
    train_env: TradingPlatform
    eval_envs: Dict[str, TradingPlatform]
    if env_type == "stock":
        train_env, eval_envs = generate_envs(generate_stock_assets())
    elif env_type == "zigzag":
        train_env, eval_envs = generate_envs(generate_zigzag_assets("1980-01-01", 5))
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model = DQN(
        "MultiInputPolicy", train_env,
        gamma=0.9,
        policy_kwargs={"net_arch": [128, 128, 128]},
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


if __name__ == "__main__":
    train("stock")
