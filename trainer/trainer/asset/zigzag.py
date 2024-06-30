import datetime
from enum import Enum
from typing import List, Tuple

import numpy as np

from .base import DailyAsset, DailyCandle


class TrendType(Enum):
    STAGNANT = 0
    UP = 1
    DOWN = 2


class Zigzag(DailyAsset):
    _published_date: datetime.date
    _published_price: float
    _trend_type_weights: Tuple[float, float, float]  # (STAGNANT, UP, DOWN)
    _stagnant_period_range: Tuple[int, int]
    _stagnant_movement_range: Tuple[float, float]  # Ratio, exclusive end
    _active_period_range: Tuple[int, int]
    _active_movement_magnitude_range: Tuple[float, float]  # Positive ratio, exclusive end
    _fluctuation_range: Tuple[float, float]  # Ratio, exclusive end

    _MA_WINDOW_RANGE: Tuple[int, int] = (5, 20)  # TODO: Choose a better window length

    def __init__(
        self,
        symbol: str,
        published_date: datetime.date,
        published_price: float,
        trend_type_weights: Tuple[float, float, float],
        stagnant_period_range: Tuple[int, int],
        stagnant_movement_range: Tuple[float, float],
        active_period_range: Tuple[int, int],
        active_movement_magnitude_range: Tuple[float, float],
        fluctuation_range: Tuple[float, float],
    ) -> None:
        super().__init__(symbol)
        if (
            (stagnant_period_range[0] > stagnant_period_range[1] or stagnant_period_range[0] < 0)
            or (stagnant_movement_range[0] > 0 or stagnant_movement_range[1] < 0)
            or (active_period_range[0] > active_period_range[1] or active_period_range[0] < 0)
            or (active_movement_magnitude_range[0] < 0 or active_movement_magnitude_range[1] < 0)
            or (fluctuation_range[0] > 0 or fluctuation_range[1] < 0)
        ):
            raise ValueError
        self._published_date = published_date
        self._published_price = published_price
        self._trend_type_weights = trend_type_weights
        self._stagnant_period_range = stagnant_period_range
        self._stagnant_movement_range = stagnant_movement_range
        self._active_period_range = active_period_range
        self._active_movement_magnitude_range = active_movement_magnitude_range
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
                trend_type = np.random.choice(
                    [TrendType.STAGNANT, TrendType.UP, TrendType.DOWN],
                    p=self._trend_type_weights,
                )
                # Number of days for next trend
                trend_days = 0
                if trend_type == TrendType.STAGNANT:
                    trend_days = np.random.randint(*self._stagnant_period_range)
                else:
                    trend_days = np.random.randint(*self._active_period_range)
                # Movement for next trend
                movement_ratio = 1.0
                if trend_type == TrendType.STAGNANT:
                    movement_ratio += np.random.uniform(*self._stagnant_movement_range)
                else:
                    movement_magnitude_ratio = trend_days * np.random.uniform(*self._active_movement_magnitude_range)
                    movement_ratio += (1 if trend_type == TrendType.UP else -1) * movement_magnitude_ratio
                # Move to next trend
                trend_start_date = trend_end_date
                trend_start_price = trend_end_price
                if (last_date - date).days < self._active_period_range[1]:
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

    def _fetch_spot_price(self) -> float:
        # TODO: Implement this method
        raise NotImplementedError


def smoothen(array: List[float], ma_window: int) -> List[float]:
    # See: https://stackoverflow.com/a/34387987
    array = np.cumsum(np.insert(array, 0, 0))
    smooth_array = (array[ma_window:] - array[:-ma_window]) / ma_window
    return smooth_array
