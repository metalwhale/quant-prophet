import datetime
import math
from typing import Callable, List, Tuple

import numpy as np

from .base import DailyAsset, DailyCandle


OPENING_HOUR: int = 8
CLOSING_HOUR: int = 16


class Sinusoid(DailyAsset):
    _published_time: datetime.datetime
    _get_price: Callable[[datetime.datetime], float]

    def __init__(
        self,
        symbol: str,
        published_time: datetime.datetime,
        alpha: float = math.pi, beta: float = 1.0, gamma1: float = 0.0, gamma2: float = 0.0,
    ) -> None:
        super().__init__(symbol)
        self._published_time = published_time
        self._get_price = lambda time: sine(time, self._published_time, alpha, beta, gamma1, gamma2)
        self._initialize()  # For fetching candles

    def _fetch_candles(self) -> List[DailyCandle]:
        date = self._published_time.date()
        if self._published_time.hour >= CLOSING_HOUR:
            date += datetime.timedelta(days=1)
        candles: List[DailyCandle] = []
        while date <= datetime.datetime.now().date():
            hourly_prices: float = [
                self._get_price(datetime.datetime.combine(date, datetime.time(hour)))
                for hour in range(OPENING_HOUR, CLOSING_HOUR + 1)
            ]
            candles.append(DailyCandle(date, max(hourly_prices), min(hourly_prices), hourly_prices[-1]))
            date += datetime.timedelta(days=1)
        return candles


class ComposedSinusoid(Sinusoid):
    def __init__(
        self,
        symbol: str,
        published_time: datetime.datetime,
        components_num: int,
        alpha_range: Tuple[float, float], beta_range: Tuple[float, float],
        gamma1_range: Tuple[float, float], gamma2_range: Tuple[float, float],
    ) -> None:
        DailyAsset.__init__(self, symbol)
        self._published_time = published_time
        components = []
        for _ in range(components_num):
            components.append(self._generate_sine(alpha_range, beta_range, gamma1_range, gamma2_range))
        self._get_price = lambda time: sum([c(time) for c in components])
        self._initialize()  # For fetching candles

    def _generate_sine(
        self,
        alpha_range: Tuple[float, float], beta_range: Tuple[float, float],
        gamma1_range: Tuple[float, float], gamma2_range: Tuple[float, float],
    ) -> Callable[[datetime.datetime], float]:
        alpha = np.random.uniform(*alpha_range)
        beta = np.random.uniform(*beta_range)
        gamma1 = np.random.uniform(*gamma1_range)
        gamma2 = np.random.uniform(*gamma2_range)
        return lambda time: sine(time, self._published_time, alpha, beta, gamma1, gamma2)


@staticmethod
# sin(α*x+γ1)*β+γ2
def sine(
    time: datetime.datetime, published_time: datetime.datetime,
    alpha: float, beta: float, gamma1: float, gamma2: float,
) -> float:
    # Use 1 day (86400 seconds) as 1 unit of time.
    x = (time - published_time).total_seconds() / 86400
    return math.sin(alpha * x + gamma1) * beta + gamma2
