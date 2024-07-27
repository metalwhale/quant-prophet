import datetime
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..asset.base import DailyAsset


class AssetDateRange:
    _asset: DailyAsset
    _date_range: List[datetime.date]

    def __init__(self, asset: DailyAsset) -> None:
        self._asset = asset
        self._date_range = []

    @property
    def asset(self) -> DailyAsset:
        return self._asset

    @property
    def date_range(self) -> List[datetime.date]:
        return self._date_range

    @date_range.setter
    def date_range(self, date_range: List[datetime.date]):
        self._date_range = date_range


class AssetPool:
    _asset_date_ranges: OrderedDict[str, AssetDateRange]
    _primary_symbols: List[str]
    _secondary_asset_generator: Optional[Callable[[], List[DailyAsset]]]

    _date_range: Tuple[Optional[datetime.date], Optional[datetime.date]]
    _historical_days_num: int
    _excluding_historical: bool

    def __init__(
        self,
        primary_assets: List[DailyAsset],
        secondary_asset_generator: Optional[Callable[[], List[DailyAsset]]] = None,
    ) -> None:
        # Use `OrderedDict` to keep secondary symbols in order, popping the oldest when renewing
        self._asset_date_ranges = OrderedDict([(a.symbol, AssetDateRange(a)) for a in primary_assets])
        self._primary_symbols = [a.symbol for a in primary_assets]
        self._secondary_asset_generator = secondary_asset_generator

    def apply_date_range(
        self,
        date_range: Tuple[Optional[datetime.date], Optional[datetime.date]],
        historical_days_num: int,
        excluding_historical: bool = True,
    ):
        self._date_range = date_range
        self._historical_days_num = historical_days_num
        self._excluding_historical = excluding_historical
        # Match date range
        min_date, max_date = date_range
        for asset_date_range in self._asset_date_ranges.values():
            asset = asset_date_range.asset
            # Choose a tradable date range
            asset_date_range.date_range = asset.find_matched_tradable_date_range(
                historical_days_num,
                min_date=min_date, max_date=max_date,
                excluding_historical=excluding_historical,
            )

    def renew_secondary_assets(self):
        if not self.is_secondary_generatable:
            return
        # Generate new secondary assets
        assets = self._secondary_asset_generator()
        # Delete old secondary assets
        for secondary_symbol in [
            s for s in self._asset_date_ranges.keys()
            if s not in self._primary_symbols  # Avoid accidentally deleting primary symbols
        ][:len(assets)]:
            self._asset_date_ranges.pop(secondary_symbol)
        # Apply date range to new secondary assets
        min_date, max_date = self._date_range
        for asset in assets:
            asset_date_range = AssetDateRange(asset)
            asset_date_range.date_range = asset.find_matched_tradable_date_range(
                self._historical_days_num,
                min_date=min_date, max_date=max_date,
                excluding_historical=self._excluding_historical,
            )
            self._asset_date_ranges[asset.symbol] = asset_date_range

    def choose_asset_date(
        self,
        favorite_symbols: Optional[List[str]] = None,
        randomizing_start: bool = False,
        preferring_secondary: bool = False,
    ) -> Tuple[str, List[datetime.date]]:
        candidate_symbols = [
            s for s in self._asset_date_ranges.keys()
            if (
                (preferring_secondary and s not in self._primary_symbols)
                or (not preferring_secondary and s in self._primary_symbols)
            ) and (favorite_symbols is None or s in favorite_symbols)
        ]
        if len(candidate_symbols) == 0:
            raise ValueError
        symbol = np.random.choice(candidate_symbols)
        # Get tradable date range
        date_range = self._asset_date_ranges[symbol].date_range
        if randomizing_start:
            # Date range needs to have at least two dates, one for the reset and one for a single step,
            # so when choosing the start date, we need to exclude the last date of date range.
            date_range = date_range[np.random.randint(len(date_range) - 1):]
        return symbol, date_range

    def get_asset(self, symbol: str) -> DailyAsset:
        return self._asset_date_ranges[symbol].asset

    @property
    def primary_symbols(self) -> List[str]:
        return self._primary_symbols

    @property
    def is_secondary_generatable(self) -> bool:
        return self._secondary_asset_generator is not None
