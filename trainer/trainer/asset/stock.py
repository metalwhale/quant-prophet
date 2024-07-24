import csv
import datetime
import os
from typing import List, Optional

from .base import DailyAsset, DailyCandle


class Stock(DailyAsset):
    _input_dir_path: os.PathLike
    _max_days_num: int

    def __init__(self, symbol: str, input_dir_path: os.PathLike, max_days_num: Optional[int] = None) -> None:
        super().__init__(symbol)
        self._input_dir_path = input_dir_path
        self._max_days_num = max_days_num
        self._initialize()  # For fetching candles

    def _fetch_candles(self) -> List[DailyCandle]:
        candles: List[DailyCandle] = []
        with open(os.path.join(self._input_dir_path, f"{self.symbol}.csv")) as stock_file:
            stock_reader = csv.DictReader(stock_file)
            for row in stock_reader:
                if row["high"] == "" or row["low"] == "" or row["close"] == "":
                    # Ignore missing data
                    continue
                date: datetime.date
                if " " in row["date"]:
                    date = datetime.datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S%z") \
                        .astimezone(datetime.timezone.utc).date()
                else:
                    date = datetime.datetime.strptime(row["date"], "%Y-%m-%d").date()
                candles.append(DailyCandle(date, float(row["high"]), float(row["low"]), float(row["close"])))
        if self._max_days_num is not None and self._max_days_num < len(candles):
            candles = candles[-self._max_days_num:]
        return candles


class StockIndex:
    _symbols: List[str]

    def __init__(self, constituents_file_path: os.PathLike, input_dir_path: os.PathLike) -> None:
        self._symbols = []
        with open(constituents_file_path) as constituents_file:
            constituents_reader = csv.DictReader(constituents_file)
            for row in constituents_reader:
                symbol = row["Symbol"]
                if os.path.isfile(os.path.join(input_dir_path, f"{symbol}.csv")):
                    self._symbols.append(symbol)

    @property
    def symbols(self) -> List[str]:
        return self._symbols
