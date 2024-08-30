import numpy as np

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
        env._position_net_price_type = np.random.choice([PriceType.ACTUAL, PriceType.MODIFIED])
        env._position_amount_type = np.random.choice([AmountType.UNIT, AmountType.SPOT])
        _, (platform_earning, calculated_earning, *_), _ = env.trade(
            max_step=np.random.randint(10, 120), rendering=False,
        )
        if abs(platform_earning - calculated_earning) >= EPSILON:
            print(platform_earning, calculated_earning)
            return False
    return True


if __name__ == "__main__":
    print(test_earning_calculation())
