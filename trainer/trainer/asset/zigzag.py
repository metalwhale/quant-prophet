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
    _trend_movement_dist: Tuple[float, float]  # Positive location and scale
    _fluctuation_range: Tuple[float, float]  # Ratio, exclusive end

    _SMA_LENGTH: int = 5  # TODO: Choose a better length

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
        # Generate raw prices
        trend_type: TrendType
        trend_end_date = self._published_date
        date = trend_end_date
        raw_price = self._published_price
        prices: List[Tuple[datetime.date, float]] = []
        sma_window: List[float] = []
        while date <= datetime.datetime.now().date() + datetime.timedelta(days=-1):
            # Smooth using SMA
            sma_window.append(raw_price)
            if len(sma_window) > self._SMA_LENGTH:
                sma_window = sma_window[len(sma_window) - self._SMA_LENGTH:]  # Remove oldest prices
            price = sum(sma_window) / len(sma_window)
            prices.append((date, price))
            if date == trend_end_date:  # Next trend
                trend_end_date += datetime.timedelta(days=np.random.randint(*self._trend_period_range))
                trend_type = np.random.choice([TrendType.UP, TrendType.DOWN], p=self._trend_type_weights)
            # Next date
            date += datetime.timedelta(days=1)
            movement_magnitude: float
            while True:
                movement_magnitude = np.random.laplace(*self._trend_movement_dist)
                if movement_magnitude >= 0:
                    break
            raw_price *= 1.0 + (1 if trend_type == TrendType.UP else -1) * movement_magnitude
        # Generate candles
        candles: List[DailyCandle] = []
        for date, price in prices:
            high = price * (1 + np.random.uniform(0, self._fluctuation_range[1]))
            low = price * (1 + np.random.uniform(self._fluctuation_range[0], 0))
            close = np.random.uniform(low, high)
            candles.append(DailyCandle(date, high, low, close))
        return candles
