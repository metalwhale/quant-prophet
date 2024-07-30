import datetime
from abc import ABC, abstractmethod
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


class DailyIndicator:
    _date: datetime.date
    _actual_price: float
    _emas: Tuple[float, float]
    _rsi: float
    _adx: float
    _cci: float

    def __init__(
        self,
        date: datetime.date,
        actual_price: float,
        emas: Tuple[float, float],
        rsi: float, adx: float, cci: float,
    ) -> None:
        self._date = date
        self._actual_price = actual_price
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


class DailyPrice:
    _date: datetime.date
    _actual_price: float
    _smoothed_price: float
    _price_delta_ratio: float  # Change in price expressed as a ratio compared to the previous day
    _ema_diff_ratio: float  # Difference in ratio between fast EMA and slow EMA
    _scaled_rsi: float
    _scaled_adx: float
    _scaled_cci: float

    def __init__(
        self,
        date: datetime.date,
        actual_price: float, smoothed_price: float,
        price_delta_ratio: float, ema_diff_ratio: float,
        scaled_rsi: float, scaled_adx: float, scaled_cci: float,
    ) -> None:
        self._date = date
        self._actual_price = actual_price
        self._smoothed_price = smoothed_price
        self._price_delta_ratio = price_delta_ratio
        self._ema_diff_ratio = ema_diff_ratio
        self._scaled_rsi = scaled_rsi
        self._scaled_adx = scaled_adx
        self._scaled_cci = scaled_cci

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
    __candles: List[DailyCandle]
    __indicators: List[DailyIndicator]
    __date_indices: Dict[str, int]

    # NOTE: When adding a new hyperparameter to calculate historical and prospective data,
    # remember to modify `calc_buffer_days_num` method
    # TODO: Choose better values
    __SMOOTHED_RADIUS = 2
    __DELTA_DISTANCE = 1
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

    def prepare_indicators(self, close_random_radius: Optional[int] = None):
        self.__indicators = []
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []
        for i, candle in enumerate(self.__candles):
            low = candle.low
            high = candle.high
            close = candle.close
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
                close = np.random.uniform(low, high)
            highs.append(high)
            lows.append(low)
            closes.append(close)
        highs = pd.Series(highs)
        lows = pd.Series(lows)
        closes = pd.Series(closes)
        for (candle, close, fast_ema, slow_ema, rsi, adx, cci) in zip(
            self.__candles,
            closes,
            EMAIndicator(closes, window=self.__EMA_WINDOW_FAST).ema_indicator(),
            EMAIndicator(closes, window=self.__EMA_WINDOW_SLOW).ema_indicator(),
            RSIIndicator(closes, window=self.__RSI_WINDOW).rsi(),
            ADXIndicator(highs, lows, closes, window=self.__ADX_WINDOW).adx(),
            CCIIndicator(highs, lows, closes, window=self.__CCI_WINDOW).cci(),
            strict=True,
        ):
            self.__indicators.append(DailyIndicator(candle.date, close, (fast_ema, slow_ema), rsi, adx, cci))

    # Returns prices within a specified date range, defined by an end date and the number of days to retrieve.
    # The actual price used is usually the close price, except for end date,
    # where there is an option to be randomly chosen within the range of low price to high price.
    def retrieve_historical_prices(self, end_date: datetime.date, days_num: int) -> List[DailyPrice]:
        historical_buffer_days_num, prospective_buffer_days_num = self.calc_buffer_days_num()
        end_date_index = self.__get_date_index(end_date)
        if (
            end_date_index is None
            # We need `days_num` days for historical data (including the end date),
            # plus a few buffer days to calculate price delta ratios and indicators for the start day.
            or end_date_index < (days_num - 1) + historical_buffer_days_num
            # We need a few buffer days of prospective data to calculate smoothed prices
            or end_date_index > (len(self.__indicators) - 1) - prospective_buffer_days_num
        ):
            raise ValueError
        prices: List[DailyPrice] = []
        # Historical prices for the days before `end_date`
        start_date_index = end_date_index - (days_num - 1)
        for i in range(start_date_index, end_date_index + 1):
            prices.append(DailyPrice(
                self.__indicators[i].date,
                self.__indicators[i].actual_price,
                sum([
                    ind.actual_price
                    for ind in self.__indicators[i - self.__SMOOTHED_RADIUS:i + self.__SMOOTHED_RADIUS + 1]
                ]) / (2 * self.__SMOOTHED_RADIUS + 1),
                self.__indicators[i].actual_price / self.__indicators[i - self.__DELTA_DISTANCE].actual_price - 1,
                self.__indicators[i].emas[0] / self.__indicators[i].emas[1] - 1,
                self.__indicators[i].rsi / 100,  # RSI range between 0 and 100
                self.__indicators[i].adx / 100,  # ADX range between 0 and 100
                self.__indicators[i].cci / 400,  # TODO: Choose a better bound for CCI
            ))
        return prices

    @classmethod
    def calc_buffer_days_num(cls) -> Tuple[int, int]:
        # TODO: Check if these buffers are properly selected
        historical_buffer_days_num = max(
            cls.__SMOOTHED_RADIUS,  # For smoothed prices
            cls.__DELTA_DISTANCE,  # For price delta ratios
            max(cls.__EMA_WINDOW_FAST - 1, cls.__EMA_WINDOW_SLOW - 1),  # For EMA diff ratios' first `nan`s
            cls.__RSI_WINDOW - 1,  # For RSI's first `nan`s
            cls.__ADX_WINDOW * 2 - 1,  # For ADX's first `0`s
            cls.__CCI_WINDOW - 1,  # For CCI's first `nan`s
        )
        prospective_buffer_days_num = cls.__SMOOTHED_RADIUS
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
