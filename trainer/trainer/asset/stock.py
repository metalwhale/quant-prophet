import csv
import datetime
import os
from typing import List, Optional

from .base import DailyAsset, DailyCandle


class Stock(DailyAsset):
    _data_dir_path: os.PathLike
    _max_days_num: int

    def __init__(self, symbol: str, stock_dir_path: os.PathLike, max_days_num: Optional[int] = None) -> None:
        super().__init__(symbol)
        self._data_dir_path = stock_dir_path
        self._max_days_num = max_days_num
        self._initialize()  # For fetching candles

    def _fetch_candles(self) -> List[DailyCandle]:
        candles: List[DailyCandle] = []
        with open(os.path.join(self._data_dir_path, f"{self.symbol}.csv")) as stock_file:
            stock_reader = csv.DictReader(stock_file)
            for row in stock_reader:
                candles.append(DailyCandle(
                    datetime.datetime.strptime(row["date"].split(" ")[0], "%Y-%m-%d").date(),
                    float(row["high"]), float(row["low"]), float(row["close"]),
                ))
        if self._max_days_num is not None and self._max_days_num < len(candles):
            candles = candles[-self._max_days_num:]
        return candles

    def _fetch_spot_price(self) -> float:
        # TODO: Implement this method
        raise NotImplementedError
