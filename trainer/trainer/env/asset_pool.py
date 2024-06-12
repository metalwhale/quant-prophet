import datetime
import random
from typing import Dict, List, Optional, Tuple

from ..asset.base import DailyAsset


class AssetPool:
    favorite_symbols: Optional[List[str]]

    _assets: Dict[str, DailyAsset]

    def __init__(self, assets: List[DailyAsset]) -> None:
        self.favorite_symbols = None
        self._assets = {asset.symbol: asset for asset in assets}

    def choose_asset(
        self,
        historical_days_num: int,
        min_date: Optional[datetime.date] = None, max_date: Optional[datetime.date] = None,
        random_start_day: bool = False,
    ) -> Tuple[str, List[datetime.date]]:
        # Choose an asset based on favorite symbols
        symbols = list(self._assets.keys())
        if self.favorite_symbols is not None:
            symbols = [symbol for symbol in self._assets.keys() if symbol in self.favorite_symbols]
        symbol = random.choice(symbols)
        # Choose a tradable date range
        date_range = self.get_asset(symbol).find_matched_tradable_date_range(
            historical_days_num,
            min_date=min_date, max_date=max_date,
        )
        if random_start_day:
            # Date range need to have at least two dates, one for the reset and one for a single step, hence `- 1`
            date_range = date_range[random.randrange(len(date_range) - 1):]
        return symbol, date_range

    def get_asset(self, symbol: str) -> DailyAsset:
        return self._assets[symbol]
