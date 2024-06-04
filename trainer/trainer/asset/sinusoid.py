import datetime
import math
import random
from typing import Callable, List, Tuple

from .base import DailyAsset, DailyCandle


class Sinusoid(DailyAsset):
    _initial_time: datetime.datetime
    _get_price: Callable[[datetime.datetime], float]

    _OPENING_HOUR: int = 8
    _CLOSING_HOUR: int = 16

    def __init__(
        self,
        initial_time: datetime.datetime,
        alpha: float = math.pi, beta: float = 1.0, gamma1: float = 0.0, gamma2: float = 0.0,
    ) -> "Sinusoid":
        self._initial_time = initial_time
        self._get_price = lambda time: self._sine(time, self._initial_time, alpha, beta, gamma1, gamma2)
        self.initialize()  # For fetching candles

    # sin(α*x+γ1)*β+γ2
    @staticmethod
    def _sine(
        time: datetime.datetime, initial_time: datetime.datetime,
        alpha: float, beta: float, gamma1: float, gamma2: float,
    ) -> float:
        # Use 1 day (86400 seconds) as 1 unit of time.
        x = (time - initial_time).total_seconds() / 86400
        return math.sin(alpha * x + gamma1) * beta + gamma2

    def _fetch_candles(self) -> List[DailyCandle]:
        date = self._initial_time.date()
        if self._initial_time.hour >= self._CLOSING_HOUR:
            date += datetime.timedelta(days=1)
        candles: List[DailyCandle] = []
        while date <= datetime.datetime.now().date():
            hourly_prices: float = [
                self._get_price(datetime.datetime.combine(date, datetime.time(hour)))
                for hour in range(self._OPENING_HOUR, self._CLOSING_HOUR + 1)
            ]
            candles.append(DailyCandle(date, max(hourly_prices), min(hourly_prices), hourly_prices[-1]))
            date += datetime.timedelta(days=1)
        return candles

    def _fetch_spot_price(self) -> float:
        return self._get_price(datetime.datetime.now())


class ComposedSinusoid(Sinusoid):
    def __init__(
        self,
        initial_time: datetime.datetime,
        components_num: int,
        alpha_range: Tuple[float, float], beta_range: Tuple[float, float],
        gamma1_range: Tuple[float, float], gamma2_range: Tuple[float, float],
    ) -> "ComposedSinusoid":
        self._initial_time = initial_time
        components = []
        for _ in range(components_num):
            components.append(self._generate_sine(alpha_range, beta_range, gamma1_range, gamma2_range))
        self._get_price = lambda time: sum([c(time) for c in components])
        self.initialize()  # For fetching candles

    def _generate_sine(
        self,
        alpha_range: Tuple[float, float], beta_range: Tuple[float, float],
        gamma1_range: Tuple[float, float], gamma2_range: Tuple[float, float],
    ) -> Callable[[datetime.datetime], float]:
        alpha = alpha_range[0] + random.random() * (alpha_range[1] - alpha_range[0])
        beta = beta_range[0] + random.random() * (beta_range[1] - beta_range[0])
        gamma1 = gamma1_range[0] + random.random() * (gamma1_range[1] - gamma1_range[0])
        gamma2 = gamma2_range[0] + random.random() * (gamma2_range[1] - gamma2_range[0])
        return lambda time: self._sine(time, self._initial_time, alpha, beta, gamma1, gamma2)
