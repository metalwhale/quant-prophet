import datetime

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator

from trainer.asset.base import DailyAsset, detect_levels, simplify
from trainer.env.asset_pool import AssetPool
from trainer.env.trading_platform import AmountType, PriceType, TradingPlatform
from train import generate_zigzag_assets


def test_earning_calculation() -> bool:
    PUBLISHED_DATE_STR = "2010-01-01"
    HISTORICAL_DAYS_NUM = 90
    EPSILON = 1e-11
    asset_pool = AssetPool(generate_zigzag_assets(PUBLISHED_DATE_STR, 100))
    asset_pool.apply_date_range((None, None), HISTORICAL_DAYS_NUM)
    env = TradingPlatform(asset_pool, HISTORICAL_DAYS_NUM)
    env._randomizing = True
    for i in range(1000):
        if i % 100 == 0:
            print(i)
        env._position_net_price_type = np.random.choice([PriceType.ACTUAL, PriceType.SIMPLIFIED])
        env._position_amount_type = np.random.choice([AmountType.UNIT, AmountType.SPOT])
        _, (platform_earning, calculated_earning, *_), _ = env.trade(
            max_step=np.random.randint(10, 120), rendering=False,
        )
        if abs(platform_earning - calculated_earning) >= EPSILON:
            print(platform_earning, calculated_earning)
            return False
    return True


def test_indicator_recalculation() -> bool:
    PUBLISHED_DATE_STR = "2020-01-01"
    HISTORICAL_DAYS_NUM = 90
    EPSILON = 1e-14
    today = datetime.datetime.today().date()
    end_date = datetime.datetime.strptime("2022-01-01", "%Y-%m-%d").date()
    while end_date < today:  # Zigzag asset fetches the candles up to yesterday. Ref: `Zigzag._fetch_candles` method.
        DailyAsset._DailyAsset__MIN_PRICE_CHANGE_RATIO_MAGNITUDE = np.random.uniform(0, 1)
        asset = generate_zigzag_assets(PUBLISHED_DATE_STR, 1)[0]
        asset.prepare_indicators()
        prices = asset.retrieve_historical_prices(end_date, HISTORICAL_DAYS_NUM,)
        # Manually calculate indicators
        end_date_index = asset._DailyAsset__get_date_index(end_date)
        actual_prices = [p.actual_price for p in asset._DailyAsset__indicators][:end_date_index + 1]
        levels = detect_levels(actual_prices, DailyAsset._DailyAsset__MIN_PRICE_CHANGE_RATIO_MAGNITUDE)
        simplified_prices = pd.Series(simplify(actual_prices, levels=levels))
        fast_emas = EMAIndicator(simplified_prices, window=DailyAsset._DailyAsset__EMA_WINDOW_FAST).ema_indicator()
        slow_emas = EMAIndicator(simplified_prices, window=DailyAsset._DailyAsset__EMA_WINDOW_SLOW).ema_indicator()
        # Compare the calculated results of indicators
        start_date_index = end_date_index - (HISTORICAL_DAYS_NUM - 1)
        fast_emas = fast_emas[start_date_index:]
        slow_emas = slow_emas[start_date_index:]
        for price, fast_ema, slow_ema in zip(prices, fast_emas, slow_emas, strict=True):
            if abs(price.ema_diff_ratio - (fast_ema / slow_ema - 1)) >= EPSILON:
                return False
        end_date += datetime.timedelta(days=1)
    return True


if __name__ == "__main__":
    print(test_earning_calculation())
    print(test_indicator_recalculation())
