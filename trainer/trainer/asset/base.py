import datetime
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import BPoly
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


class DailyIndicator:
    _date: datetime.date
    _actual_price: float
    _modified_price: float
    _emas: Tuple[float, float]
    _rsi: float
    _adx: float
    _cci: float

    def __init__(
        self,
        date: datetime.date, actual_price: float, modified_price: float,
        emas: Tuple[float, float], rsi: float, adx: float, cci: float,
    ) -> None:
        self._date = date
        self._actual_price = actual_price
        self._modified_price = modified_price
        self._emas = emas
        self._rsi = rsi
        self._adx = adx
        self._cci = cci

    @property
    def date(self) -> datetime.date:
        return self._date

    @property
    def actual_price(self) -> float:
        return self._actual_price

    @property
    def modified_price(self) -> float:
        return self._modified_price

    @property
    def emas(self) -> Tuple[float, float]:
        return self._emas

    @property
    def rsi(self) -> float:
        return self._rsi

    @property
    def adx(self) -> float:
        return self._adx

    @property
    def cci(self) -> float:
        return self._cci


class LevelType(Enum):
    SUPPORT = 0
    RESISTANCE = 1


class DailyPrice:
    _date: datetime.date
    _level_type: Optional[LevelType]
    _actual_price: float
    _modified_price: float
    _price_delta_ratio: float  # Change in price expressed as a ratio compared to the previous day
    _ema_diff_ratio: float  # Difference in ratio between fast EMA and slow EMA
    _scaled_rsi: float
    _scaled_adx: float
    _scaled_cci: float

    def __init__(
        self,
        date: datetime.date,
        level_type: Optional[LevelType], actual_price: float, modified_price: float,
        price_delta_ratio: float, ema_diff_ratio: float,
        scaled_rsi: float, scaled_adx: float, scaled_cci: float,
    ) -> None:
        self._date = date
        self._level_type = level_type
        self._actual_price = actual_price
        self._modified_price = modified_price
        self._price_delta_ratio = price_delta_ratio
        self._ema_diff_ratio = ema_diff_ratio
        self._scaled_rsi = scaled_rsi
        self._scaled_adx = scaled_adx
        self._scaled_cci = scaled_cci

    @property
    def date(self) -> datetime.date:
        return self._date

    @property
    def level_type(self) -> Optional[LevelType]:
        return self._level_type

    @property
    def actual_price(self) -> float:
        return self._actual_price

    @property
    def modified_price(self) -> float:
        return self._modified_price

    @property
    def price_delta_ratio(self) -> float:
        return self._price_delta_ratio

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
    # Initialized only once
    __candles: List[DailyCandle]
    __date_indices: Dict[str, int]
    # Recalculated when preparing indicator values
    __levels: OrderedDict[int, LevelType]
    __indicators: List[DailyIndicator]

    # TODO: Choose better values
    __LEVEL_DETECTION_PRICE_CHANGE: Optional[Tuple[Tuple[float, float], ...]] = None
    __LEVEL_CONCATENATION_PULLBACK_DAYS_NUM: int = 0
    # NOTE: When adding a new hyperparameter to calculate historical and futuristic data,
    # remember to modify `calc_buffer_days_num` method
    __DELTA_DISTANCE = 1
    __EMA_WINDOW_FAST = 5
    __EMA_WINDOW_SLOW = 20
    __RSI_WINDOW = 14
    __ADX_WINDOW = 14
    __CCI_WINDOW = 20
    __RETRACTION_DISTANCE = 0

    __DATE_FORMAT = "%Y-%m-%d"

    def __init__(self, symbol: str) -> None:
        self.__symbol = symbol

    # Find the widest date range that matches the following conditions:
    # - Chosen from the days of `self.__candles`
    # - All dates must be equal to or greater than `min_date`
    # - Skip the first few days which are reserved for calculating historical data
    # - All dates must be equal to or smaller than `max_date`
    # - Skip the last few days which are reserved for calculating futuristic data
    def find_matched_tradable_date_range(
        self,
        historical_days_num: int,
        min_date: Optional[datetime.date] = None, max_date: Optional[datetime.date] = None,
        excluding_historical: bool = True,
    ) -> List[datetime.date]:
        historical_buffer_days_num, futuristic_buffer_days_num = self.calc_buffer_days_num()
        if (
            (min_date is not None and max_date is not None and min_date > max_date)
            or len(self.__candles) < historical_days_num + historical_buffer_days_num + futuristic_buffer_days_num
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
            if i > (len(self.__candles) - 1) - futuristic_buffer_days_num:
                break
            date_range.append(date)
        return date_range

    def prepare_indicators(self, random_radius: Optional[int] = None):
        highs = []
        lows = []
        closes = []
        for i, candle in enumerate(self.__candles):
            low: float
            high: float
            close: float
            if random_radius is None:
                low = candle.low
                high = candle.high
                close = candle.close
            else:
                j = max(i - random_radius, 0)
                min_low = self.__candles[j].low
                max_low = min_low
                min_high = self.__candles[j].high
                max_high = min_high
                # Iterate through the neighboring candles within a specific radius,
                # where a `radius = 0` means using only the current candle (no neighbors).
                while j < min(i + random_radius, len(self.__candles) - 1):
                    j += 1
                    min_low = min(min_low, self.__candles[j].low)
                    max_low = max(max_low, self.__candles[j].low)
                    min_high = min(min_high, self.__candles[j].high)
                    max_high = max(max_high, self.__candles[j].high)
                # TODO: Find a better way to handle cases where the max low is greater than the min high
                if max_low > min_high:
                    max_low, min_high = min_high, max_low
                low = np.random.uniform(min_low, max_low)
                high = np.random.uniform(min_high, max_high)
                close = np.random.uniform(low, high)
            highs.append(high)
            lows.append(low)
            closes.append(close)
        # Modify the prices
        highs = pd.Series(highs)
        lows = pd.Series(lows)
        closes = pd.Series(closes)
        modified_closes: pd.Series
        if self.__LEVEL_DETECTION_PRICE_CHANGE is not None:
            self.__levels = _detect_complex_levels(
                closes, *self.__LEVEL_DETECTION_PRICE_CHANGE,
                self.__LEVEL_CONCATENATION_PULLBACK_DAYS_NUM,
            )
            modified_closes = pd.Series(_smooth(closes, self.__levels))
        else:
            self.__levels = OrderedDict()
            modified_closes = closes.copy()
        # Calculate indicators
        fast_emas = EMAIndicator(closes, window=self.__EMA_WINDOW_FAST).ema_indicator()
        slow_emas = EMAIndicator(closes, window=self.__EMA_WINDOW_SLOW).ema_indicator()
        rsis = RSIIndicator(closes, window=self.__RSI_WINDOW).rsi()
        adxs = ADXIndicator(highs, lows, closes, window=self.__ADX_WINDOW).adx()
        ccis = CCIIndicator(highs, lows, closes, window=self.__CCI_WINDOW).cci()
        # Retract the modified closes by a specific distance into the past
        for i in reversed(range(len(modified_closes))):
            if i - self.__RETRACTION_DISTANCE >= 0:
                modified_closes[i] = modified_closes[i - self.__RETRACTION_DISTANCE]
            else:
                modified_closes[i] = float("nan")
        # Store the indicators
        self.__indicators = []
        for (candle, actual_close, modified_close, fast_ema, slow_ema, rsi, adx, cci) in zip(
            self.__candles, closes, modified_closes,
            fast_emas, slow_emas, rsis, adxs, ccis,
            strict=True,
        ):
            self.__indicators.append(DailyIndicator(
                candle.date, actual_close, modified_close,
                # Indicators near the end date will be recalculated each time historical data is retrieved
                (fast_ema, slow_ema), rsi, adx, cci,
            ))

    # Returns prices within a specified date range, defined by an end date and the number of days to retrieve.
    # The actual price used is usually the close price, except for end date,
    # where there is an option to be randomly chosen within the range of low price to high price.
    def retrieve_historical_prices(self, end_date: datetime.date, days_num: int) -> List[DailyPrice]:
        historical_buffer_days_num, futuristic_buffer_days_num = self.calc_buffer_days_num()
        end_date_index = self.__get_date_index(end_date)
        if (
            end_date_index is None
            # We need `days_num` days for historical data (including the end date),
            # plus a few buffer days to calculate price delta ratios and indicators for the start day.
            or end_date_index < (days_num - 1) + historical_buffer_days_num
            # We also need a few buffer days for futuristic data
            or end_date_index > (len(self.__indicators) - 1) - futuristic_buffer_days_num
        ):
            raise ValueError
        prices: List[DailyPrice] = []
        start_date_index = end_date_index - (days_num - 1)
        indicators = self.__indicators[start_date_index:end_date_index + 1]
        # Historical prices for the days before `end_date`
        for i in range(len(indicators)):
            prices.append(DailyPrice(
                indicators[i].date,
                self.__levels.get(start_date_index + i, None),
                indicators[i].actual_price,
                indicators[i].modified_price,
                # Features
                indicators[i].actual_price / indicators[i - self.__DELTA_DISTANCE].actual_price - 1,
                indicators[i].emas[0] / indicators[i].emas[1] - 1,
                indicators[i].rsi / 100,  # RSI range between 0 and 100
                indicators[i].adx / 100,  # ADX range between 0 and 100
                indicators[i].cci / 400,  # TODO: Choose a better bound for CCI
            ))
        return prices

    @classmethod
    def calc_buffer_days_num(cls) -> Tuple[int, int]:
        # TODO: Check if these buffers are properly selected
        historical_buffer_days_num = max(
            cls.__DELTA_DISTANCE,  # For price delta ratios
            max(cls.__EMA_WINDOW_FAST - 1, cls.__EMA_WINDOW_SLOW - 1),  # For first `nan`s of EMA diff ratios
            cls.__RSI_WINDOW - 1,  # For first `nan`s of RSIs
            cls.__ADX_WINDOW * 2 - 1,  # For first `0`s of ADXs
            cls.__CCI_WINDOW - 1,  # For first `nan`s of CCIs
            cls.__RETRACTION_DISTANCE,  # For first `nan`s of modified closes
        )
        futuristic_buffer_days_num = 0
        return historical_buffer_days_num, futuristic_buffer_days_num

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


def _detect_complex_levels(
    prices: "pd.Series[float]",
    # For detecting levels
    major_price_change: Tuple[float, float],  # Ratio magnitude
    minor_price_change: Tuple[float, float],  # Ratio magnitude
    significant_price_change: Tuple[float, float],  # Ratio magnitude
    # For concatenating levels
    pullback_days_num: int,
) -> OrderedDict[int, LevelType]:
    # Price change between levels
    major_support_change, major_resistance_change = major_price_change
    minor_support_change, minor_resistance_change = minor_price_change
    significant_support_change, significant_resistance_change = significant_price_change
    if (
        significant_support_change < 0 or significant_support_change > 1
        or significant_resistance_change < 0 or significant_resistance_change > 1
    ):
        # Must be in the range 0 to 1
        raise ValueError
    # Detect major levels
    major_levels = _detect_simple_levels(prices, major_price_change)
    if len(major_levels) == 0:
        return major_levels
    # Find trends with significant price change
    levels: OrderedDict[int, LevelType] = OrderedDict()
    major_level_indices = list(major_levels.keys())
    for major_index, next_major_index in zip(major_level_indices[:-1], major_level_indices[1:]):
        levels[major_index] = major_levels[major_index]
        if (
            major_levels[major_index] == LevelType.SUPPORT
            and prices[next_major_index] / prices[major_index] >= 1 + significant_resistance_change
        ) or (
            major_levels[major_index] == LevelType.RESISTANCE
            and prices[next_major_index] / prices[major_index] <= 1 - significant_support_change
        ):
            price_change: Tuple[float, float]
            if major_levels[major_index] == LevelType.SUPPORT:
                price_change = (minor_support_change, major_resistance_change)
            elif major_levels[major_index] == LevelType.RESISTANCE:
                price_change = (major_support_change, minor_resistance_change)
            else:
                raise ValueError
            # Detect minor levels
            trend_prices = prices[major_index:next_major_index + 1].copy().reset_index(drop=True)
            minor_levels = _detect_simple_levels(
                trend_prices, price_change,
                first_level_type=major_levels[next_major_index],
            )
            if (
                # The first minor level, if detected, must have the same type as the major level that starts this trend
                0 in minor_levels
                and minor_levels[0] != major_levels[major_index]
            ) or (
                # The last minor level, if detected, must have the same type as the major level that ends this trend
                len(trend_prices) - 1 in minor_levels
                and minor_levels[len(trend_prices) - 1] != major_levels[next_major_index]
            ):
                raise Exception
            minor_level_indices = list(minor_levels.keys())
            if len(minor_level_indices) > 0 and 0 not in minor_levels:
                # Skip the first minor levels if the first price change is insufficient
                minor_levels.pop(minor_level_indices[0])
                minor_levels.pop(minor_level_indices[1])
                # We don't skip the last minor levels because, in a trend with significant price changes,
                # they are nearly the same price as the last date of the trend.
                # Detecting them helps us take profit earlier rather than waiting until the last date.
            for minor_index, minor_level_type in minor_levels.items():
                levels[major_index + minor_index] = minor_level_type
    levels[major_level_indices[-1]] = major_levels[major_level_indices[-1]]
    # Concatenate levels
    levels = _concatenate_levels(levels, pullback_days_num)
    return levels


# In "Case 2" updating the last level requires checking for a new value in the second-last level.
# Initially, it might seem like that after updating the second-last level we need to apply a same rule recursively,
# and update all previous levels back to the first level, but this is not necessary.
#
# To understand this, we need to know why updating the last level might also require updating the second-last level.
# Let's imagine we have a support at the second-last level and a resistance at the last level (old index).
# When prices continue to move, we encounter a new higher value (new index).
# In this case, we want to update the last resistance level from its old index to the new index.
# A problem is that there may be a value in the range from the old index to the new index of the last resistance level
# that is lower than value of the second-last support level.
# The case where the second-last level is a resistance and the last level is a support is similar.
#
# However, we only need to update the second-last level if the last level is either support or resistance.
# The case we should handle depends on the relationship between minimum price change ratios of two level types.
# Reason is that we need to handle both cases if and only if the following conditions are both met:
# - (1): There are S1 -> R1 -> S2 in chronological order, where S1 is a support, R1 is a resistance,
#   and S2 is lower than S1 but not a support (not sufficiently lower than R1 by a specified support ratio).
#   In this case, if an R2 value appears after S2 that is higher than R1,
#   then R2 will become the new resistance and S2 will become the new support.
# - (2): Similar to (1) but with R1 -> S1 -> R2 in chronological order, where R1 is a resistance, S1 is a support,
#   and R2 is higher than R1 but not a resistance (not sufficiently higher than S1 by a specified resistance ratio).
# Let's prove that (1) and (2) cannot both be true.
#
# Denote mS as the magnitude of the minimum price change ratio for support, and mR as the magnitude for resistance.
# In case of (1), assume that value of R1 is 1 unit. Then value of S1 is < 1/(1+mR). Since S2 < S1, value of S2 is < 1/(1+mR).
# For (1) to be true, S2 must be > 1-mS so that it cannot be the support after R1.
# Combining these formulas, we have (I): 1-mS < 1/(1+mR).
# Applying to (2), we have (II): 1+mR > 1/(1-mS).
# Mathematically, (I) and (II) cannot both be true, this means we only need to update the second-last level
# if either the last level is support or resistance (depending on how we choose the ratios), but not in both cases.
# In other words, when updating the last level to a new value, the oldest levels we need to retrace are up to the second-last level.
#
# TODO:
# The following function implements "Case 2" to always update both the last and second-last levels,
# regardless of whether the last level is support or resistance.
# According to the above argument, this is redundant, and we should add checking code to avoid unnecessary handling.
def _detect_simple_levels(
    prices: "pd.Series[float]",
    price_change: Tuple[float, float],  # Ratio magnitude
    first_level_type: LevelType = LevelType.SUPPORT,
) -> OrderedDict[int, LevelType]:
    # Price change between levels
    support_change, resistance_change = price_change
    if (
        support_change < 0 or support_change > 1
        or resistance_change < 0 or resistance_change > 1
    ):
        # Must be in the range 0 to 1
        raise ValueError
    # Detect levels
    levels: OrderedDict[int, LevelType] = OrderedDict()
    index = 0
    levels[index] = first_level_type
    while index < len(prices) - 1:
        index += 1
        last1_level_index = list(levels.keys())[-1]
        last2_level_index = list(levels.keys())[-2] if len(levels) >= 2 else None
        if levels[last1_level_index] == LevelType.SUPPORT:
            # Case 1: Add a new resistance level if its price is higher than the last support level by a specified ratio
            if prices[index] / prices[last1_level_index] >= 1 + resistance_change:
                levels[index] = LevelType.RESISTANCE
            # Case 2: Update the last support level if the new price is lower
            elif prices[index] / prices[last1_level_index] <= 1:
                levels.pop(last1_level_index)
                # Find the highest price before the current price:
                # - If there is no second-last level (only the last level), we start searching from the beginning
                # - We use `[::-1]` to reverse the array so that `argmax` can find the last index of the highest price
                #   in case there are multiple occurrences of the highest price with the same value
                start_index = last1_level_index if last2_level_index is not None else 0
                highest_index = (index - 1) - np.argmax(prices[start_index:index][::-1])
                if last2_level_index is not None:
                    # Update the second-last resistance level to the new highest price
                    if prices[highest_index] / prices[last2_level_index] >= 1:
                        levels.pop(last2_level_index)
                        last2_level_index = highest_index
                        levels[last2_level_index] = LevelType.RESISTANCE
                else:
                    # Add the highest price as a second-last resistance level, preceding the last support level
                    if prices[index] / prices[highest_index] <= 1 - support_change:
                        last2_level_index = highest_index
                        levels[last2_level_index] = LevelType.RESISTANCE
                # Update the last support level
                levels[index] = LevelType.SUPPORT
        elif levels[last1_level_index] == LevelType.RESISTANCE:
            # Case 1: Add a new support level if its price is lower than the last resistance level by a specified ratio
            if prices[index] / prices[last1_level_index] <= 1 - support_change:
                levels[index] = LevelType.SUPPORT
            # Case 2: Update the last resistance level if the new price is higher
            elif prices[index] / prices[last1_level_index] >= 1:
                levels.pop(last1_level_index)
                # Find the lowest price before the current price:
                # - If there is no second-last level (only the last level), we start searching from the beginning
                # - We use `[::-1]` to reverse the array so that `argmin` can find the last index of the lowest price
                #   in case there are multiple occurrences of the lowest price with the same value
                start_index = last1_level_index if last2_level_index is not None else 0
                lowest_index = (index - 1) - np.argmin(prices[start_index:index][::-1])
                if last2_level_index is not None:
                    # Update the second-last support level to the new lowest price
                    if prices[lowest_index] / prices[last2_level_index] <= 1:
                        levels.pop(last2_level_index)
                        last2_level_index = lowest_index
                        levels[last2_level_index] = LevelType.SUPPORT
                else:
                    # Add the lowest price as a second-last support level, preceding the last resistance level
                    if prices[index] / prices[lowest_index] >= 1 + resistance_change:
                        last2_level_index = lowest_index
                        levels[last2_level_index] = LevelType.SUPPORT
                # Update the last resistance level
                levels[index] = LevelType.RESISTANCE
    if len(levels) == 1:
        # If no levels are detected other than the initial one, clear the list of levels
        levels = OrderedDict()
    return levels


def _concatenate_levels(
    levels: OrderedDict[int, LevelType],
    pullback_days_num: int,  # Max number of days to be considered a short trend
) -> OrderedDict[int, LevelType]:
    # Concatenate pullback levels
    concatenated_levels: OrderedDict[int, LevelType] = OrderedDict()
    level_indices = list(levels.keys())
    i = 0
    while i < len(level_indices):
        current_index = level_indices[i]
        is_pullback = i > 0 and current_index - level_indices[i - 1] < pullback_days_num
        if not is_pullback:
            # Only consider the current level if it is not at the end of a trend that is too short
            concatenated_levels[current_index] = levels[current_index]
        i += 1
    if len(concatenated_levels) == 1:
        # If all levels are concatenated into a single level, clear the list of levels
        concatenated_levels = OrderedDict()
    return concatenated_levels


def _smooth(prices: "pd.Series[float]", levels: OrderedDict[int, LevelType]) -> np.ndarray:
    # Treat the first and last prices as levels
    level_indices = sorted(set(levels.keys()) | {0, len(prices) - 1})
    # `0` means setting the derivative to zero at each level
    interpolate = BPoly.from_derivatives(level_indices, [[prices[i], 0] for i in level_indices])
    smoothed_prices = interpolate(np.arange(len(prices)))
    return smoothed_prices
