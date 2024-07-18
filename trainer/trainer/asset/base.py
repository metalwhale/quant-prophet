import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class DailyCandle:
    _date: datetime.date
    _high: float
    _low: float
    _close: float

    def __init__(
        self,
        date: datetime.date,
        high: float, low: float, close: float,
    ) -> None:
        self._date = date
        self._high = high
        self._low = low
        self._close = close

    @property
    def date(self) -> datetime.date:
        return self._date

    @property
    def high(self) -> float:
        return self._high

    @property
    def low(self) -> float:
        return self._low

    @property
    def close(self) -> float:
        return self._close


class DailyIndicator:
    _date: datetime.date
    _price_range: Tuple[float, float]
    _actual_price: float   # Close price or a random value between low price and high price
    _fast_ema: float
    _slow_ema: float

    def __init__(
        self,
        date: datetime.date,
        price_range: Tuple[float, float], actual_price: float,
        fast_ema: float, slow_ema: float,
    ) -> None:
        self._date = date
        self._price_range = price_range
        self._actual_price = actual_price
        self._fast_ema = fast_ema
        self._slow_ema = slow_ema

    @property
    def date(self) -> datetime.date:
        return self._date

    @property
    def price_range(self) -> Tuple[float, float]:
        return self._price_range

    @property
    def actual_price(self) -> float:
        return self._actual_price

    @property
    def fast_ema(self) -> float:
        return self._fast_ema

    @property
    def slow_ema(self) -> float:
        return self._slow_ema


class DailyPrice:
    _date: datetime.date
    _actual_price: float
    _smoothed_price: float
    _price_delta: float  # Change in price expressed as a ratio compared to the previous day
    _ema_delta: float  # Change in ema expressed as a ratio compared to the previous day
    _ema_price_diff: float  # Difference in ratio between ema and price
    _ema_diff: float  # Difference in ratio between ema fast-length and slow-length

    def __init__(
        self,
        date: datetime.date,
        actual_price: float, smoothed_price: float,
        price_delta: float, ema_delta: float, ema_price_diff: float, ema_diff: float,
    ) -> None:
        self._date = date
        self._actual_price = actual_price
        self._smoothed_price = smoothed_price
        self._price_delta = price_delta
        self._ema_delta = ema_delta
        self._ema_price_diff = ema_price_diff
        self._ema_diff = ema_diff

    @property
    def date(self) -> datetime.date:
        return self._date

    @property
    def actual_price(self) -> float:
        return self._actual_price

    @property
    def smoothed_price(self) -> float:
        return self._smoothed_price

    @property
    def price_delta(self) -> float:
        return self._price_delta

    @property
    def ema_delta(self) -> float:
        return self._ema_delta

    @property
    def ema_price_diff(self) -> float:
        return self._ema_price_diff

    @property
    def ema_diff(self) -> float:
        return self._ema_diff


class DailyAsset(ABC):
    __symbol: str
    __candles: List[DailyCandle]
    __indicators: List[DailyIndicator]
    __date_indices: Dict[str, int]

    # NOTE: When adding a new parameter for calculating indicators, remember to modify `calc_buffer_days_num` method
    __FAST_EMA_LENGTH = 5  # TODO: Choose a better length
    __SLOW_EMA_LENGTH = 20  # TODO: Choose a better length (longer than fast-length)
    __DELTA_DISTANCE = 1

    __DATE_FORMAT = "%Y-%m-%d"

    def __init__(self, symbol: str) -> None:
        self.__symbol = symbol

    # Find the widest date range that matches the following conditions:
    # - Chosen from the days of `self.__candles`
    # - All dates must be equal to or greater than `min_date`
    # - Skip the first few days which are reserved for calculating historical data
    # - All dates must be equal to or smaller than `max_date`
    def find_matched_tradable_date_range(
        self,
        historical_days_num: int,
        min_date: Optional[datetime.date] = None, max_date: Optional[datetime.date] = None,
        excluding_historical: bool = True,
    ) -> List[datetime.date]:
        if (
            (min_date is not None and max_date is not None and min_date > max_date)
            or len(self.__candles) < historical_days_num + self.calc_buffer_days_num()
        ):
            return []
        date_range: List[datetime.date] = []
        # `extra` means that in order to calculate anything (other than the actual price),
        # which requires data from previous days, we also count the days even before the first historical days.
        extra_historical_days_count = 0
        for candle in self.__candles:
            date = candle.date
            if min_date is not None and date < min_date:
                continue
            extra_historical_days_count += 1
            # Stop skipping when we reach the last day of historical days
            if (
                excluding_historical
                and extra_historical_days_count < historical_days_num + self.calc_buffer_days_num()
            ):
                continue
            if max_date is not None and date > max_date:
                break
            date_range.append(date)
        return date_range

    # TODO: Use `self.__indicators` instead of `self.__candles` when retrieving price deltas
    def retrieve_price_delta(self, date: datetime.date) -> Optional[float]:
        date_index = self.__get_date_index(date)
        if date_index is None or date_index < self.__DELTA_DISTANCE:
            return None
        return self.__candles[date_index].close / self.__candles[date_index - self.__DELTA_DISTANCE].close - 1

    def prepare_indicators(self, close_random_radius: Optional[int] = None):
        self.__indicators = []
        prev_fast_ema: Optional[float] = None
        prev_slow_ema: Optional[float] = None
        for i, candle in enumerate(self.__candles):
            low = candle.low
            high = candle.high
            actual_price = candle.close
            if close_random_radius is not None:
                j = max(i - close_random_radius, 0)
                low = self.__candles[j].low
                high = self.__candles[j].high
                # Iterate through the neighboring candles within a specific radius,
                # where a `radius = 0` means using only the current candle (no neighbors).
                while j < min(i + close_random_radius, len(self.__candles) - 1):
                    j += 1
                    low = min(low, self.__candles[j].low)
                    high = max(high, self.__candles[j].high)
                actual_price = np.random.uniform(low, high)
            # See: https://en.wikipedia.org/wiki/Exponential_smoothing
            fast_ema = calc_ema(actual_price, prev_fast_ema, self.__FAST_EMA_LENGTH)
            slow_ema = calc_ema(actual_price, prev_slow_ema, self.__SLOW_EMA_LENGTH)
            self.__indicators.append(DailyIndicator(candle.date, (low, high), actual_price, fast_ema, slow_ema))
            prev_fast_ema = fast_ema
            prev_slow_ema = slow_ema

    # Returns prices within a specified date range, defined by an end date and the number of days to retrieve.
    # The actual price used is usually the close price, except for end date,
    # where there is an option to be randomly chosen within the range of low price to high price.
    def retrieve_historical_prices(
        self, end_date: datetime.date, days_num: int,
        randomizing_end: bool = True,
    ) -> List[DailyPrice]:
        end_index = self.__get_date_index(end_date)
        # We need `days_num` days for historical data (including the end date),
        # plus a few buffer days to calculate price deltas and indicators for the start day.
        if end_index is None or end_index < (days_num - 1) + self.calc_buffer_days_num():
            return []
        end_indicator = self.__indicators[end_index]
        prices: List[DailyPrice] = []
        # Historical prices for the days before `end_date`
        start_date_index = end_index - (days_num - 1)
        for i in range(start_date_index, end_index):
            prices.append(DailyPrice(
                self.__indicators[i].date,
                self.__indicators[i].actual_price,
                self.__indicators[i].fast_ema,
                self.__indicators[i].actual_price / self.__indicators[i - self.__DELTA_DISTANCE].actual_price - 1,
                self.__indicators[i].slow_ema / self.__indicators[i - self.__DELTA_DISTANCE].slow_ema - 1,
                self.__indicators[i].actual_price / self.__indicators[i].slow_ema - 1,
                self.__indicators[i].fast_ema / self.__indicators[i].slow_ema - 1,
            ))
        # Price for `end_date`
        end_actual_price: float
        if randomizing_end:
            end_actual_price = np.random.uniform(*end_indicator.price_range)
        else:
            end_actual_price = end_indicator.actual_price
        end_fast_ema = calc_ema(
            end_actual_price, self.__indicators[end_index - 1].fast_ema,
            self.__FAST_EMA_LENGTH,
        )
        end_slow_ema = calc_ema(
            end_actual_price, self.__indicators[end_index - 1].slow_ema,
            self.__SLOW_EMA_LENGTH,
        )
        prices.append(DailyPrice(
            end_indicator.date,
            end_actual_price,
            end_fast_ema,
            end_actual_price / self.__indicators[end_index - self.__DELTA_DISTANCE].actual_price - 1,
            end_slow_ema / self.__indicators[end_index - self.__DELTA_DISTANCE].slow_ema - 1,
            end_actual_price / end_slow_ema - 1,
            end_fast_ema / end_slow_ema - 1,
        ))
        return prices

    @classmethod
    def calc_buffer_days_num(cls) -> int:
        # TODO: Add more buffer for MAs
        return (
            # For `DailyIndicator` MAs
            max(cls.__FAST_EMA_LENGTH, cls.__SLOW_EMA_LENGTH)
            # For `DailyPrice` deltas
            + cls.__DELTA_DISTANCE
        )

    @property
    def symbol(self) -> str:
        return self.__symbol

    # Remember to call this method in the inheritance class to fetch candles
    def _initialize(self):
        self.__candles = self._fetch_candles()
        self.__candles.sort(key=lambda candle: candle.date)  # Sort the list just in case
        self.__date_indices = {
            c.date.strftime(self.__DATE_FORMAT): i
            for i, c in enumerate(self.__candles)
        }

    # Returns list of candles in ascending order of day
    @abstractmethod
    def _fetch_candles(self) -> List[DailyCandle]:
        pass

    def __get_date_index(self, date: datetime.date) -> Optional[int]:
        return self.__date_indices.get(date.strftime(self.__DATE_FORMAT))


def calc_ema(price: float, prev_ema: Optional[float], length: int) -> float:
    if prev_ema is None:
        prev_ema = price
    ema_smoothing_factor = 2 / (length + 1)  # See: https://www.investopedia.com/terms/e/ema.asp
    ema = ema_smoothing_factor * price + (1 - ema_smoothing_factor) * prev_ema
    return ema
