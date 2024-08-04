import datetime
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator


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


class PriceType(Enum):
    ACTUAL = 0
    SIMPLIFIED = 1


class DailyPrice:
    _date: datetime.date
    _actual_price: float
    _simplified_price: float
    _ema_diff_ratio: float  # Difference in ratio between fast EMA and slow EMA
    _scaled_rsi: float
    _scaled_adx: float
    _scaled_cci: float

    def __init__(
        self,
        date: datetime.date,
        actual_price: float, simplified_price: float,
        ema_diff_ratio: float,
        scaled_rsi: float, scaled_adx: float, scaled_cci: float,
    ) -> None:
        self._date = date
        self._actual_price = actual_price
        self._simplified_price = simplified_price
        self._ema_diff_ratio = ema_diff_ratio
        self._scaled_rsi = scaled_rsi
        self._scaled_adx = scaled_adx
        self._scaled_cci = scaled_cci

    def get_price(self, price_type: PriceType) -> float:
        if price_type == PriceType.ACTUAL:
            return self.actual_price
        elif price_type == PriceType.SIMPLIFIED:
            return self.simplified_price
        else:
            raise NotImplementedError

    @property
    def date(self) -> datetime.date:
        return self._date

    @property
    def actual_price(self) -> float:
        return self._actual_price

    @property
    def simplified_price(self) -> float:
        return self._simplified_price

    @property
    def ema_diff_ratio(self) -> float:
        return self._ema_diff_ratio

    @property
    def scaled_rsi(self) -> float:
        return self._scaled_rsi

    @property
    def scaled_adx(self) -> float:
        return self._scaled_adx

    @property
    def scaled_cci(self) -> float:
        return self._scaled_cci


class DailyAsset(ABC):
    __symbol: str
    __candles: List[DailyCandle]
    __prepared_candles: List[DailyCandle]
    __date_indices: Dict[str, int]

    # NOTE: When adding a new hyperparameter to calculate historical and prospective data,
    # remember to modify `calc_buffer_days_num` method
    # TODO: Choose better values
    __EMA_WINDOW_FAST = 5
    __EMA_WINDOW_SLOW = 20
    __RSI_WINDOW = 14
    __ADX_WINDOW = 14
    __CCI_WINDOW = 20

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
        historical_buffer_days_num, prospective_buffer_days_num = self.calc_buffer_days_num()
        if (
            (min_date is not None and max_date is not None and min_date > max_date)
            or len(self.__candles) < historical_days_num + historical_buffer_days_num + prospective_buffer_days_num
        ):
            raise ValueError
        date_range: List[datetime.date] = []
        # `extra` means that in order to calculate anything (other than the actual price),
        # which requires data from previous days, we also count the days even before the first historical days.
        extra_historical_days_count = 0
        for i, candle in enumerate(self.__candles):
            date = candle.date
            if min_date is not None and date < min_date:
                continue
            extra_historical_days_count += 1
            # Stop skipping when we reach the last day of historical days
            if (
                excluding_historical
                and extra_historical_days_count < historical_days_num + historical_buffer_days_num
            ):
                continue
            if max_date is not None and date > max_date:
                break
            if i >= (len(self.__candles) - 1) - prospective_buffer_days_num:
                break
            date_range.append(date)
        return date_range

    def prepare_candles(self, close_random_radius: Optional[int] = None):
        self.__prepared_candles = []
        highs = []
        lows = []
        closes = []
        for i, candle in enumerate(self.__candles):
            low = candle.low
            high = candle.high
            close = candle.close
            if close_random_radius is not None:
                j = max(i - close_random_radius, 0)
                low = self.__candles[j].low
                high = self.__candles[j].high
                # Iterate through the neighbor candles within a specific radius,
                # where a `radius = 0` means using only the current candle (no neighbors).
                while j < min(i + close_random_radius, len(self.__candles) - 1):
                    j += 1
                    low = min(low, self.__candles[j].low)
                    high = max(high, self.__candles[j].high)
                close = np.random.uniform(low, high)
            highs.append(high)
            lows.append(low)
            closes.append(close)
        for candle, high, low, close in zip(self.__candles, highs, lows, closes, strict=True):
            self.__prepared_candles.append(DailyCandle(candle.date, high, low, close))

    # Returns prices within a specified date range, defined by an end date and the number of days to retrieve.
    # The actual price used is usually the close price, except for end date,
    # where there is an option to be randomly chosen within the range of low price to high price.
    def retrieve_historical_prices(
        self, end_date: datetime.date, days_num: int,
        min_price_change_ratio_magnitude: Optional[float],
    ) -> List[DailyPrice]:
        historical_buffer_days_num, prospective_buffer_days_num = self.calc_buffer_days_num()
        end_date_index = self.__get_date_index(end_date)
        if (
            end_date_index is None
            # We need `days_num` days for historical data (including the end date),
            # plus a few buffer days to calculate indicators for the start day.
            or end_date_index < (days_num - 1) + historical_buffer_days_num
            # We need a few buffer days of prospective data just in case
            or end_date_index > (len(self.__prepared_candles) - 1) - prospective_buffer_days_num
        ):
            raise ValueError
        # Simplify the prices
        dates = []
        highs = []
        lows = []
        closes = []
        for i in range(end_date_index + 1):
            dates.append(self.__prepared_candles[i].date)
            highs.append(self.__prepared_candles[i].high)
            lows.append(self.__prepared_candles[i].low)
            closes.append(self.__prepared_candles[i].close)
        simplified_highs = highs
        simplified_lows = lows
        simplified_closes = closes
        if min_price_change_ratio_magnitude is not None:
            simplified_highs = simplify(highs, min_price_change_ratio_magnitude)
            simplified_lows = simplify(lows, min_price_change_ratio_magnitude)
            simplified_closes = simplify(closes, min_price_change_ratio_magnitude)
        actual_closes = pd.Series(closes)
        simplified_highs = pd.Series(simplified_highs)
        simplified_lows = pd.Series(simplified_lows)
        simplified_closes = pd.Series(simplified_closes)
        # Calculate indicators
        fast_emas = EMAIndicator(simplified_closes, window=self.__EMA_WINDOW_FAST).ema_indicator()
        slow_emas = EMAIndicator(simplified_closes, window=self.__EMA_WINDOW_SLOW).ema_indicator()
        rsis = RSIIndicator(simplified_closes, window=self.__RSI_WINDOW).rsi()
        adxs = ADXIndicator(simplified_highs, simplified_lows, simplified_closes, window=self.__ADX_WINDOW).adx()
        ccis = CCIIndicator(simplified_highs, simplified_lows, simplified_closes, window=self.__CCI_WINDOW).cci()
        # Historical prices for the days before `end_date`
        prices: List[DailyPrice] = []
        start_date_index = end_date_index - (days_num - 1)
        for i in range(start_date_index, end_date_index + 1):
            prices.append(DailyPrice(
                dates[i],
                actual_closes[i], simplified_closes[i],
                fast_emas[i] / slow_emas[i] - 1,
                rsis[i] / 100,  # RSI range between 0 and 100
                adxs[i] / 100,  # ADX range between 0 and 100
                ccis[i] / 400,  # TODO: Choose a better bound for CCI
            ))
        return prices

    @classmethod
    def calc_buffer_days_num(cls) -> Tuple[int, int]:
        # TODO: Check if these buffers are properly selected
        historical_buffer_days_num = max(
            max(cls.__EMA_WINDOW_FAST - 1, cls.__EMA_WINDOW_SLOW - 1),  # For EMA diff ratios' first `nan`s
            cls.__RSI_WINDOW - 1,  # For RSI's first `nan`s
            cls.__ADX_WINDOW * 2 - 1,  # For ADX's first `0`s
            cls.__CCI_WINDOW - 1,  # For CCI's first `nan`s
        )
        prospective_buffer_days_num = 0
        return (historical_buffer_days_num, prospective_buffer_days_num)

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def published_date(self) -> datetime.date:
        return self.__candles[0].date

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


class LevelType(Enum):
    SUPPORT = 0
    RESISTANCE = 1


def simplify(prices: List[float], min_price_change_ratio_magnitude: float) -> List[float]:
    if min_price_change_ratio_magnitude < 0:
        # Should be positive
        raise ValueError
    # Detect levels
    level_types: OrderedDict[int, LevelType] = OrderedDict()
    index = 0
    level_types[index] = LevelType.RESISTANCE  # An arbitrary initial level, which can be set to any level type
    while index < len(prices) - 1:
        index += 1
        last1_level_index = list(level_types.keys())[-1]
        last2_level_index = list(level_types.keys())[-2] if len(level_types) >= 2 else None
        if level_types[last1_level_index] == LevelType.SUPPORT:
            # Case 1: Add a new resistance level if its price is higher than the last support level by a specified ratio
            if prices[index] / prices[last1_level_index] >= 1 + min_price_change_ratio_magnitude:
                level_types[index] = LevelType.RESISTANCE
            # Case 2: Update the last support level if the new price is lower
            elif prices[index] <= prices[last1_level_index]:
                level_types.pop(last1_level_index)
                # Find the highest price before the current price:
                # - If there is no second-last level (only the last level), we start searching from the beginning
                # - We use `[::-1]` to reverse the array so that `argmax` can find the last index of the highest price
                #   in case there are multiple occurrences of the highest price with the same value
                start_index = last1_level_index if last2_level_index is not None else 0
                highest_index = (index - 1) - np.argmax(prices[start_index:index][::-1])
                if last2_level_index is not None:
                    # Update the second-last resistance level to the new highest price
                    if prices[highest_index] >= prices[last2_level_index]:
                        level_types.pop(last2_level_index)
                        level_types[highest_index] = LevelType.RESISTANCE
                else:
                    # Add the highest price as a second-last resistance level, preceding the last support level
                    if prices[index] / prices[highest_index] <= 1 - min_price_change_ratio_magnitude:
                        level_types[highest_index] = LevelType.RESISTANCE
                # Update the last support level
                level_types[index] = LevelType.SUPPORT
        elif level_types[last1_level_index] == LevelType.RESISTANCE:
            # Case 1: Add a new support level if its price is lower than the last resistance level by a specified ratio
            if prices[index] / prices[last1_level_index] <= 1 - min_price_change_ratio_magnitude:
                level_types[index] = LevelType.SUPPORT
            # Case 2: Update the last resistance level if the new price is higher
            elif prices[index] >= prices[last1_level_index]:
                level_types.pop(last1_level_index)
                # Find the lowest price before the current price:
                # - If there is no second-last level (only the last level), we start searching from the beginning
                # - We use `[::-1]` to reverse the array so that `argmin` can find the last index of the lowest price
                #   in case there are multiple occurrences of the lowest price with the same value
                start_index = last1_level_index if last2_level_index is not None else 0
                lowest_index = (index - 1) - np.argmin(prices[start_index:index][::-1])
                if last2_level_index is not None:
                    # Update the second-last support level to the new lowest price
                    if prices[lowest_index] <= prices[last2_level_index]:
                        level_types.pop(last2_level_index)
                        level_types[lowest_index] = LevelType.SUPPORT
                else:
                    # Add the lowest price as a second-last support level, preceding the last resistance level
                    if prices[index] / prices[lowest_index] >= 1 + min_price_change_ratio_magnitude:
                        level_types[lowest_index] = LevelType.SUPPORT
                # Update the last resistance level
                level_types[index] = LevelType.RESISTANCE
    if len(level_types) == 1:
        # When no levels are detected other than the initial one, we clear the level list
        level_types = OrderedDict()
    # Simplify the prices using the detected level indices
    level_indices = list(level_types.keys())
    if len(level_indices) == 0 or level_indices[0] != 0:
        level_indices.insert(0, 0)
    if len(level_indices) == 0 or level_indices[-1] != len(prices) - 1:
        level_indices.append(len(prices) - 1)
    simplified_prices: List[float] = []
    for index, next_index in zip(level_indices[:len(level_indices) - 1], level_indices[1:]):
        for i in range(index, next_index):
            simplified_price = prices[index] + (prices[next_index] - prices[index]) * (i - index) / (next_index - index)
            simplified_prices.append(simplified_price)
    simplified_prices.append(prices[-1])
    return simplified_prices
