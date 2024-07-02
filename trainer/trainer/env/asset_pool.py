import datetime
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..asset.base import DailyAsset


class DatePolarity:
    _date: datetime.date
    _diff: int

    def __init__(self, date: datetime.date, diff: int) -> None:
        self._date = date
        self._diff = diff

    @property
    def date(self) -> datetime.date:
        return self._date

    @property
    def diff(self) -> int:
        return self._diff


class AssetDateRange:
    _asset: DailyAsset
    _date_polarities: List[DatePolarity]

    def __init__(self, asset: DailyAsset) -> None:
        self._asset = asset
        self._date_polarities = []

    @property
    def asset(self) -> DailyAsset:
        return self._asset

    @property
    def date_polarities(self) -> List[DatePolarity]:
        return self._date_polarities

    @date_polarities.setter
    def date_polarities(self, date_polarities: List[DatePolarity]):
        self._date_polarities = date_polarities


class AssetPool:
    favorite_symbols: Optional[List[str]]

    _asset_date_ranges: Dict[str, AssetDateRange]
    _polarity_temperature: float

    def __init__(self, assets: List[DailyAsset], polarity_temperature: float = 1.0) -> None:
        self.favorite_symbols = None
        self._asset_date_ranges = {a.symbol: AssetDateRange(a) for a in assets}
        self._polarity_temperature = polarity_temperature

    def apply_date_range_matcher(
        self,
        ahead_days_num: int,
        historical_days_num: int,
        date_range: Tuple[Optional[datetime.date], Optional[datetime.date]],
        excluding_historical: bool = True,
    ):
        min_date, max_date = date_range
        for asset_date_range in self._asset_date_ranges.values():
            asset = asset_date_range.asset
            # Choose a tradable date range
            tradable_date_range = asset.find_matched_tradable_date_range(
                historical_days_num,
                min_date=min_date, max_date=max_date,
                excluding_historical=excluding_historical,
            )
            asset_date_range.date_polarities = _map_to_date_polarity(asset, tradable_date_range, ahead_days_num)

    def choose_asset_date(
        self,
        randomizing_start: bool = False,
        target_polarity_diff: Optional[int] = None,
    ) -> Tuple[str, List[datetime.date]]:
        # Choose an asset based on favorite symbols
        symbols = list(self._asset_date_ranges.keys())
        if self.favorite_symbols is not None:
            symbols = [s for s in self._asset_date_ranges.keys() if s in self.favorite_symbols]
        symbol = np.random.choice(symbols)
        # Get tradable date range
        date_polarities = self._asset_date_ranges[symbol].date_polarities
        date_range = [p.date for p in date_polarities]
        if randomizing_start:
            # Date range needs to have at least two dates, one for the reset and one for a single step,
            # so when choosing the start date, we need to exclude the last date of date range.
            start_date_index = 0
            if target_polarity_diff is None:
                # Exclude the last date of date range with `- 1`
                start_date_index = np.random.randint(len(date_range) - 1)
            else:
                # Exclude the last date of date range with `-1`
                polarity_distances = [abs(p.diff - target_polarity_diff) for p in date_polarities[:-1]]
                # Sort the dates by distance to the target polarity diff in descending order.
                # By doing this, we ensure that dates with a closer distance to the target polarity diff have a greater weight.
                distances = sorted(set(polarity_distances), reverse=True)
                # `+ 1` because indices are 0-based, but we want the weights to be greater than 0
                distance_weights = [distances.index(d) + 1 for d in polarity_distances]
                start_date_index = np.random.choice(
                    range(len(distance_weights)),
                    p=_softmax(distance_weights, self._polarity_temperature),
                )
            date_range = date_range[start_date_index:]
        return symbol, date_range

    def get_asset(self, symbol: str) -> DailyAsset:
        return self._asset_date_ranges[symbol].asset


def calc_polarity_diff(price_delta: float) -> int:
    return 1 if price_delta >= 0 else -1


def _map_to_date_polarity(
    asset: DailyAsset,
    date_range: List[datetime.date], ahead_days_num: int,
) -> List[DatePolarity]:
    asset_polarities: List[DatePolarity] = []
    diff = 0
    # First, we iterate over all dates in the date range and calculate the cumulative polarity diff,
    # so that it is the sum of at most `ahead_days_num + 1` polarity diffs (`ahead_days_num` dates before and 1 current date).
    for i in range(len(date_range)):
        diff += calc_polarity_diff(asset.retrieve_price_delta(date_range[i]))
        if i >= ahead_days_num:
            # Assign the sum of polarity diffs to the date `ahead_days_num` days ago
            date = date_range[i - ahead_days_num]
            asset_polarities.append(DatePolarity(date, diff))
            # Remove the oldest diff before moving to the next date
            diff -= calc_polarity_diff(asset.retrieve_price_delta(date))
    # Finally, we iterate through the remaining dates within the `ahead_days_num` date range from the last date
    for i in range(max(0, len(date_range) - ahead_days_num), len(date_range)):
        date = date_range[i]
        asset_polarities.append(DatePolarity(date, diff))
        # Remove the oldest diff before moving to the next date
        diff -= calc_polarity_diff(asset.retrieve_price_delta(date))
    return asset_polarities


def _softmax(array: List, temperature: float) -> List:
    exp = [math.exp(e / temperature) for e in array]
    exp_sum = sum(exp)
    return [e / exp_sum for e in exp]
