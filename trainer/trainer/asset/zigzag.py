import datetime
from enum import Enum
from typing import List, Tuple

import numpy as np

from .base import DailyAsset, DailyCandle


class TrendType(Enum):
    UP = 1
    DOWN = 2


class Zigzag(DailyAsset):
    _published_date: datetime.date
    _published_price: float
    _trend_type_weights: Tuple[float, float]  # (UP, DOWN)
    _trend_period_range: Tuple[int, int]
    _trend_movement_dist: Tuple[float, float]  # Positive mean and standard deviation
    _fluctuation_range: Tuple[float, float]  # Ratio, exclusive end

    _MA_WINDOW_RANGE: Tuple[int, int] = (5, 20)  # TODO: Choose a better window length

    def __init__(
        self,
        symbol: str,
        published_date: datetime.date,
        published_price: float,
        trend_type_weights: Tuple[float, float],
        trend_period_range: Tuple[int, int],
        trend_movement_dist: Tuple[float, float],
        fluctuation_range: Tuple[float, float],
    ) -> None:
        super().__init__(symbol)
        if (
            (trend_period_range[0] > trend_period_range[1] or trend_period_range[0] < 0)
            or (trend_movement_dist[0] < 0 or trend_movement_dist[1] < 0)
            or (fluctuation_range[0] > 0 or fluctuation_range[1] < 0)
        ):
            raise ValueError
        self._published_date = published_date
        self._published_price = published_price
        self._trend_type_weights = trend_type_weights
        self._trend_period_range = trend_period_range
        self._trend_movement_dist = trend_movement_dist
        self._fluctuation_range = fluctuation_range
        self._initialize()  # For fetching candles

    def _fetch_candles(self) -> List[DailyCandle]:
        trend_start_date: datetime.date
        trend_start_price: float
        trend_end_date = self._published_date
        trend_end_price = self._published_price
        date = trend_end_date
        last_date = datetime.datetime.now().date() + datetime.timedelta(days=-1)
        # Generate raw prices
        raw_prices: List[Tuple[datetime.date, float]] = []
        while date <= last_date:
            price = 0.0
            if date == trend_end_date:
                # Trend type for next trend
                trend_type = np.random.choice([TrendType.UP, TrendType.DOWN], p=self._trend_type_weights)
                # Number of days for next trend
                trend_days = np.random.randint(*self._trend_period_range)
                # Movement for next trend
                movement_magnitude_ratio = trend_days * max(0, np.random.normal(*self._trend_movement_dist))
                movement_ratio = 1.0 + (1 if trend_type == TrendType.UP else -1) * movement_magnitude_ratio
                # Move to next trend
                trend_start_date = trend_end_date
                trend_start_price = trend_end_price
                if (last_date - date).days < self._trend_period_range[1]:
                    trend_end_date = last_date
                else:
                    trend_end_date += datetime.timedelta(days=trend_days)
                trend_end_price *= movement_ratio
                price = trend_start_price
            else:
                price = trend_start_price \
                    + (date - trend_start_date).days / (trend_end_date - trend_start_date).days \
                    * (trend_end_price - trend_start_price)
            date += datetime.timedelta(days=1)
            raw_prices.append((date, price))
        # Generate candles
        candles: List[DailyCandle] = []
        for date, price in zip(
            [d for d, _ in raw_prices],
            smoothen([p for _, p in raw_prices], np.random.randint(*self._MA_WINDOW_RANGE)),
        ):
            high = price * (1 + np.random.uniform(0, self._fluctuation_range[1]))
            low = price * (1 + np.random.uniform(self._fluctuation_range[0], 0))
            close = np.random.uniform(low, high)
            candles.append(DailyCandle(date, high, low, close))
        return candles


def smoothen(array: List[float], ma_window: int) -> List[float]:
    # See: https://stackoverflow.com/a/34387987
    array = np.cumsum(np.insert(array, 0, 0))
    smooth_array = (array[ma_window:] - array[:-ma_window]) / ma_window
    return smooth_array
