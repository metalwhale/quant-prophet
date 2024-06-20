import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

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


class DailyPrice:
    _date: datetime.date
    _actual_price: float  # Close price or a random value between low price and high price
    _price_delta: float  # Change in price expressed as a ratio compared to the previous day

    def __init__(self, date: datetime.date, actual_price: float, price_delta: float) -> None:
        self._date = date
        self._actual_price = actual_price
        self._price_delta = price_delta

    @property
    def date(self) -> datetime.date:
        return self._date

    @property
    def actual_price(self) -> float:
        return self._actual_price

    @property
    def price_delta(self) -> float:
        return self._price_delta


class DailyAsset(ABC):
    __symbol: str
    __org_candles: List[DailyCandle]
    __candles: List[DailyCandle]
    __date_indices: Dict[str, int]

    __DATE_FORMAT = "%Y-%m-%d"
    __DELTA_DISTANCE = 1

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
        exclude_historical: bool = True,
    ) -> List[datetime.date]:
        if (
            (min_date is not None and max_date is not None and min_date > max_date)
            # Price delta requires `self.__DELTA_DISTANCE` days to calculate
            or len(self.__org_candles) < historical_days_num + self.__DELTA_DISTANCE
        ):
            return []
        date_range: List[datetime.date] = []
        # `extra` means that in order to calculate anything (other than the actual price),
        # which requires data from previous days, we also count the days even before the first historical days.
        extra_historical_days_count = 0
        for candle in self.__org_candles:
            date = candle.date
            if min_date is not None and date < min_date:
                continue
            extra_historical_days_count += 1
            # Stop skipping when we reach the last day of historical days
            if exclude_historical and extra_historical_days_count < historical_days_num + self.__DELTA_DISTANCE:
                continue
            if max_date is not None and date > max_date:
                break
            date_range.append(date)
        return date_range

    # TODO: Use `self.__candles` instead of `self.__org_candles` when retrieving price deltas
    def retrieve_price_delta(self, date: datetime.date) -> float:
        date_index = self._get_date_index(date)
        return self.__org_candles[date_index].close / self.__org_candles[date_index - self.__DELTA_DISTANCE].close - 1

    def prepare_candles(self, random_close: bool = False):
        self.__candles = [
            DailyCandle(
                c.date, c.high, c.low,
                c.low + np.random.random() * (c.high - c.low) if random_close else c.close,
            )
            for c in self.__org_candles
        ]

    # Returns prices within a specified date range, defined by an end date and the number of days to retrieve.
    # The actual price used for calculating price delta is usually the close price, except for end date,
    # where there is an option to be randomly chosen within the range of low price to high price.
    def retrieve_historical_prices(
        self, end_date: datetime.date, days_num: int,
        random_end: bool = True,
    ) -> List[DailyPrice]:
        end_date_index = self._get_date_index(end_date)
        today = datetime.datetime.now().date()
        end_candle = self.__candles[end_date_index]
        if (
            # We need `days_num` days for historical data (including the end date),
            # plus `self.__DELTA_DISTANCE` to calculate price delta for the start day.
            end_date_index is None or end_date_index < days_num + self.__DELTA_DISTANCE - 1
            or end_candle.date > today
        ):
            return []
        prices: List[DailyPrice] = []
        # Historical prices for the days before `end_date`
        start_date_index = end_date_index - (days_num - 1)
        for i in range(start_date_index, end_date_index):
            prices.append(DailyPrice(
                self.__candles[i].date,
                self.__candles[i].close,
                self.__candles[i].close / self.__candles[i - self.__DELTA_DISTANCE].close - 1,
            ))
        # Price for `end_date`
        end_date_price = 0
        if end_date == today:
            end_date_price = self._fetch_spot_price()
        else:
            if random_end:
                end_date_price = end_candle.low + np.random.random() * (end_candle.high - end_candle.low)
            else:
                end_date_price = end_candle.close
        prices.append(DailyPrice(
            end_candle.date,
            end_date_price,
            end_date_price / self.__candles[end_date_index - self.__DELTA_DISTANCE].close - 1,
        ))
        return prices

    @property
    def symbol(self) -> str:
        return self.__symbol

    # Remember to call this method in the inheritance class to fetch candles
    def _initialize(self):
        self.__org_candles = self._fetch_candles()
        self.__org_candles.sort(key=lambda candle: candle.date)  # Sort the list just in case
        self.__date_indices = {
            c.date.strftime(self.__DATE_FORMAT): i
            for i, c in enumerate(self.__org_candles)
        }

    def _get_date_index(self, date: datetime.date) -> int:
        return self.__date_indices.get(date.strftime(self.__DATE_FORMAT))

    # Returns list of candles in ascending order of day
    @abstractmethod
    def _fetch_candles(self) -> List[DailyCandle]:
        pass

    # Returns the price at the current moment
    @abstractmethod
    def _fetch_spot_price(self) -> float:
        pass
