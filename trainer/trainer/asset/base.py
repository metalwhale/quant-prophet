import datetime
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class DailyCandle:
    _date: datetime.date
    _high: float
    _low: float
    _close: float

    def __init__(
        self,
        date: datetime.date,
        high: float, low: float, close: float,
    ) -> "DailyCandle":
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

    def __init__(self, date: datetime.date, actual_price: float, price_delta: float) -> "DailyPrice":
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
    __candles: List[DailyCandle]
    __candle_indices: Dict[str, int]

    __DATE_FORMAT = "%Y-%m-%d"

    # Find the widest date range that matches the following conditions:
    # - Chosen from the days of `self.__candles`
    # - All dates must be equal to or greater than `min_date`
    # - Skip the first `historical_days_num` days and maintain the order
    #   (More precisely, we skip `historical_days_num - 1 + 1` days,
    #    where `- 1` indicates stopping skipping when we reach the end date, and `+ 1` indicates the day before the start day).
    # - All dates must be equal to or smaller than `max_date`
    def find_matched_tradable_date_range(
        self,
        historical_days_num: int,
        min_date: Optional[datetime.date] = None, max_date: Optional[datetime.date] = None,
    ) -> List[datetime.date]:
        if (
            (min_date is not None and max_date is not None and min_date > max_date)
            # We need at least `historical_days_num` plus 1 because price delta requires one previous day to calculate
            or len(self.__candles) < historical_days_num + 1
        ):
            return []
        date_range: List[datetime.date] = []
        historical_days_count = 0
        for candle in self.__candles:
            date = candle.date
            if min_date is not None and date < min_date:
                continue
            historical_days_count += 1
            if historical_days_count <= historical_days_num:
                continue
            if max_date is not None and date > max_date:
                break
            date_range.append(date)
        return date_range

    # Returns prices within a specified date range, defined by an end date and the number of days to retrieve.
    # The actual price used for calculating price delta is usually the close price, except for end date,
    # where there is an option to be randomly chosen within the range of low price to high price.
    def retrieve_historical_prices(
        self, end_date: datetime.date, days_num: int,
        indeterministic: bool = True,
    ) -> List[DailyPrice]:
        end_date_index = self.__candle_indices.get(end_date.strftime(self.__DATE_FORMAT))
        # The end date needs to be at least at `days_num` index
        # because we need `days_num` days for historical data (including the end date),
        # plus one day to calculate price delta for the start day.
        today = datetime.datetime.now().date()
        if (
            end_date_index is None or end_date_index < days_num
            or self.__candles[end_date_index].date > today
        ):
            return []
        prices: List[DailyPrice] = []
        # Historical prices for the days before `end_date`
        for i in range(end_date_index - (days_num - 1), end_date_index):
            prices.append(DailyPrice(
                self.__candles[i].date,
                self.__candles[i].close,
                self.__candles[i].close / self.__candles[i - 1].close - 1,
            ))
        # Price for `end_date`
        end_date_price = 0
        if end_date == today:
            end_date_price = self._fetch_spot_price()
        else:
            if indeterministic:
                end_date_price = self.__candles[end_date_index].low \
                    + random.random() * (self.__candles[end_date_index].high - self.__candles[end_date_index].low)
            else:
                end_date_price = self.__candles[end_date_index].close
        prices.append(DailyPrice(
            self.__candles[end_date_index].date,
            end_date_price,
            (end_date_price / self.__candles[end_date_index - 1].close - 1)
        ))
        return prices

    # Remember to call this method in the inheritance class to fetch candles
    def _initialize(self):
        self.__candles = self._fetch_candles()
        self.__candles.sort(key=lambda candle: candle.date)  # Sort the list just in case
        self.__candle_indices = {
            c.date.strftime(self.__DATE_FORMAT): i
            for i, c in enumerate(self.__candles)
        }

    # Returns list of candles in ascending order of day
    @abstractmethod
    def _fetch_candles(self) -> List[DailyCandle]:
        pass

    # Returns the price at the current moment
    @abstractmethod
    def _fetch_spot_price(self) -> float:
        pass
