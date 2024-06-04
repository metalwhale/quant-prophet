import datetime
import math
import random
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Union

from .util import OPENING_HOUR, CLOSING_HOUR, find_min_tradable_start_date


class Price:
    _time: datetime.datetime
    _actual_price: float  # Actual price of an asset at a specific time
    _price_delta: float  # Change in price expressed as a ratio compared to the previous day

    def __init__(self, time: datetime, actual_price: float, price_delta: float) -> "Price":
        self._time = time
        self._actual_price = actual_price
        self._price_delta = price_delta

    @property
    def time(self) -> datetime.datetime:
        return self._time

    @property
    def actual_price(self) -> float:
        return self._actual_price

    @property
    def price_delta(self) -> float:
        return self._price_delta


class DailyAsset(ABC):
    @abstractmethod
    def get_first_date(self) -> datetime.date:
        pass

    @abstractmethod
    # This method returns prices within a specified date range, defined by an end date and the number of days to retrieve.
    # Actual price used for calculating price delta is usually the close price except for end date,
    # where it should be randomized within the range of low price to high price if time is not specified but only date.
    def retrieve_historical_prices(
        self,
        end_datetime: Union[datetime.date, datetime.datetime], days_num: int,
    ) -> List[Price]:
        pass


class Sinusoid(DailyAsset):
    _initial_time: datetime.datetime
    _get_price: Callable[[datetime.datetime], float]

    def __init__(
        self,
        initial_time: datetime.datetime,
        alpha: float = math.pi, beta: float = 1.0,
        gamma1: float = 0.0, gamma2: float = 0,
    ) -> "Sinusoid":
        self._initial_time = initial_time
        self._get_price = lambda time: self._sine(time, self._initial_time, alpha, beta, gamma1, gamma2)

    def get_first_date(self) -> datetime.date:
        return find_min_tradable_start_date(self._initial_time)

    def retrieve_historical_prices(
        self,
        end_datetime: Union[datetime.date, datetime.datetime], days_num: int,
    ) -> List[Price]:
        if days_num < 1:
            return []
        end_date: datetime.date = end_datetime.date() if type(end_datetime) == datetime.datetime else end_datetime
        start_date = end_date + datetime.timedelta(days=-(days_num - 1))
        if start_date < self.get_first_date():
            return []
        start_close_time = datetime.datetime(start_date.year, start_date.month, start_date.day, hour=CLOSING_HOUR)
        end_open_time = datetime.datetime(end_date.year, end_date.month, end_date.day, hour=OPENING_HOUR)
        end_close_time = datetime.datetime(end_date.year, end_date.month, end_date.day, hour=CLOSING_HOUR)
        prices: List[Price] = []
        # Retrieve prices before end date.
        prev_time = start_close_time + datetime.timedelta(days=-1)
        cur_time = start_close_time
        while cur_time.date() < end_close_time.date():
            prices.append(self._calculate_price(prev_time, cur_time))
            prev_time = cur_time
            cur_time += datetime.timedelta(days=1)
        # Retrieve price for end date, with current price randomly chosen between open and close times.
        cur_time: datetime.datetime
        if type(end_datetime) == datetime.datetime:
            # TODO: Raise an error if the end datetime is greater than close time (i.e. outside of daily trading hours).
            # It's ok for now because we use `datetime` type only for rendering (see `TradingPlatform` class, `render` method),
            # where the last datetime is randomly chosen by this `retrieve_historical_prices` method.
            cur_time = min(end_datetime, end_close_time)
        else:
            cur_time = end_open_time + \
                datetime.timedelta(seconds=random.randint(0, int((end_close_time - end_open_time).total_seconds())))
        prices.append(self._calculate_price(prev_time, cur_time))
        return prices

    def _calculate_price(self, prev_time: datetime.datetime, cur_time: datetime.datetime) -> Price:
        cur_price = self._get_price(cur_time)
        return Price(
            cur_time,
            cur_price,
            cur_price / self._get_price(prev_time) - 1
        )

    # sin(α*x+γ1)*β+γ2
    @staticmethod
    def _sine(
        time: datetime.datetime, initial_time: datetime.datetime,
        alpha: float, beta: float, gamma1: float, gamma2: float,
    ) -> float:
        # Use 1 day (86400 seconds) as 1 unit of time.
        x = (time - initial_time).total_seconds() / 86400
        return math.sin(alpha * x + gamma1) * beta + gamma2


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
