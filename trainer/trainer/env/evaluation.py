import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3.common.callbacks import BaseCallback

from .trading_platform import TradingPlatform


class FullEvalCallback(BaseCallback):
    _output_path: Path
    _envs: Dict[str, TradingPlatform]
    _freq: int
    _action_diff_threshold: float
    _showing_image: bool

    _ep_count: int
    _avg_overview_results: List[Dict[str, Any]]
    _asset_overview_results: Dict[str, List[Dict[str, Any]]]

    _OVERVIEW_LOG_FILE_NAME = "overview.csv"
    _OVERVIEW_CHART_FILE_NAME = "overview.png"

    def __init__(
        self,
        output_path: Path, envs: Dict[str, TradingPlatform], freq: int,
        action_diff_threshold: float = 0.0, showing_image: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        # Parameters
        self._output_path = output_path
        self._envs = envs
        self._freq = freq
        self._action_diff_threshold = action_diff_threshold
        self._showing_image = showing_image
        # Initialization
        os.makedirs(output_path, exist_ok=True)
        self._ep_count = 0
        self._avg_overview_results = []
        self._asset_overview_results = defaultdict(list)

    def _on_training_start(self) -> None:
        self.__eval_model()
        return super()._on_training_start()

    def _on_step(self) -> bool:
        # See: https://github.com/DLR-RM/stable-baselines3/blob/v2.3.2/stable_baselines3/common/callbacks.py#L590-L631
        if np.sum(self.locals["dones"]).item() != 0:
            self._ep_count += 1
            if self._ep_count > 0 and self._ep_count % self._freq == 0:
                self.__eval_model()
        return super()._on_step()

    def _on_training_end(self) -> None:
        self.__eval_model()
        return super()._on_training_end()

    def __eval_model(self):
        avg_result: Dict[str, Any] = {"ep_count": self._ep_count}
        asset_results: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"ep_count": self._ep_count})
        for env_name, env in self._envs.items():
            env.set_mode(False)  # Just in case
            earning_discrepancy_sum = 0.0
            wl_rate_sum = 0.0
            for symbol in env.asset_pool.primary_symbols:
                os.makedirs(self._output_path / symbol, exist_ok=True)
                env.favorite_symbols = [symbol]
                (
                    rendered,
                    (_, earning, price_change, wl_rate),
                    (positions, last_price),
                ) = env.trade(
                    model=self.model,
                    action_diff_threshold=self._action_diff_threshold, stopping_when_done=False,
                )
                earning_discrepancy = self.__calc_earning_discrepancy(price_change, earning)
                asset_results[symbol] |= {
                    "ep_count": self._ep_count,
                    f"{env_name}_earning_discrepancy": earning_discrepancy,
                    f"{env_name}_wl_rate": wl_rate,
                }
                # Write trade positions
                with open(self._output_path / symbol / f"trade_{env_name}_{self._ep_count}.csv", "w") as positions_file:
                    positions_writer = csv.DictWriter(positions_file, ["date", "entry_price", "position_type"])
                    positions_writer.writeheader()
                    for position in positions:
                        positions_writer.writerow({
                            "date": position.date,
                            "entry_price": position.entry_price.actual_price,
                            "position_type": position.position_type.name,
                        })
                    positions_writer.writerow({
                        "date": last_price.date,
                        "entry_price": last_price.actual_price,
                    })
                # Draw trade chart
                image = Image.fromarray(rendered)
                draw = ImageDraw.Draw(image)
                draw.text((60, 60), f"env={env_name}, episode={self._ep_count}:\n" + ", ".join([
                    f"earning={earning:.2f}",
                    f"price_change={price_change:.2f}",
                    f"wl_rate={wl_rate:.2f}",
                ]), fill=(0, 0, 0), font_size=80)
                if self._showing_image:
                    plt.close("all")
                    show_image(image)
                image.save(self._output_path / symbol / f"trade_{env_name}_{self._ep_count}.png")
                # For average overview
                earning_discrepancy_sum += earning_discrepancy
                wl_rate_sum += wl_rate
            avg_result |= {
                f"{env_name}_earning_discrepancy": earning_discrepancy_sum / len(env.asset_pool.primary_symbols),
                f"{env_name}_wl_rate": wl_rate_sum / len(env.asset_pool.primary_symbols),
            }
            env.favorite_symbols = None
        # Save overview results
        for symbol in set([s for e in self._envs.values() for s in e.asset_pool.primary_symbols]):
            self.__save_overview_results(symbol, asset_results[symbol])
        self.__save_overview_results(None, avg_result)

    def __save_overview_results(self, asset_symbol: Optional[str], result: Dict[str, Any]):
        overview_results: List[Dict[str, Any]]
        overview_log_file_path: Path
        overview_chart_file_path: Path
        if asset_symbol is None:
            overview_results = self._avg_overview_results
            overview_log_file_path = self._output_path / self._OVERVIEW_LOG_FILE_NAME
            overview_chart_file_path = self._output_path / self._OVERVIEW_CHART_FILE_NAME
        else:
            overview_results = self._asset_overview_results[asset_symbol]
            overview_log_file_path = self._output_path / asset_symbol / self._OVERVIEW_LOG_FILE_NAME
            overview_chart_file_path = self._output_path / asset_symbol / self._OVERVIEW_CHART_FILE_NAME
        # Write overview log
        overview_log_field_names = [
            "ep_count",
            *[f for n in self._envs.keys() for f in [f"{n}_earning_discrepancy", f"{n}_wl_rate"]],
        ]
        if not os.path.isfile(overview_log_file_path):
            with open(overview_log_file_path, "w") as overview_log_file:
                overview_log_writer = csv.DictWriter(overview_log_file, overview_log_field_names)
                overview_log_writer.writeheader()
                overview_log_file.flush()
        with open(overview_log_file_path, "a") as overview_log_file:
            overview_log_writer = csv.DictWriter(overview_log_file, overview_log_field_names)
            overview_log_writer.writerow({k: v for k, v in result.items() if k in overview_log_field_names})
            overview_log_file.flush()
        # Draw overview chart
        overview_results.append(result)
        # LINK: `num` and `clear` help prevent memory leak (See: https://stackoverflow.com/a/65910539)
        figure = plt.figure(num="evaluation", figsize=(10, 3 * len(self._envs)), dpi=400, clear=True)
        figure.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        for i, env_name in enumerate(self._envs.keys()):
            axes = figure.add_subplot(len(self._envs), 1, i + 1)
            ep_counts = [r["ep_count"] for r in overview_results]
            axes.set_title(env_name)
            axes.plot(ep_counts, [0 for _ in overview_results], color="gray")
            axes.plot(ep_counts, [r[f"{env_name}_earning_discrepancy"] for r in overview_results], color="blue")
            axes.plot(ep_counts, [r[f"{env_name}_wl_rate"] for r in overview_results], color="orange")
        figure.canvas.draw()
        image = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8) \
            .reshape(figure.canvas.get_width_height()[::-1] + (3,))
        if self._showing_image:
            plt.close("all")
        Image.fromarray(image).save(overview_chart_file_path)

    @staticmethod
    def __calc_earning_discrepancy(price_change: float, earning: float) -> float:
        # Increase earning by the absolute value of it,
        # plus double the absolute value of the price change if they have opposite signs.
        magnitude = abs(earning) + (0 if earning * price_change >= 0 else 2 * abs(price_change))
        magnitude_ratio = magnitude / abs(price_change) - 1
        # Take into account the sign of the original earning value
        return (1 if earning >= 0 else -1) * magnitude_ratio


def show_image(image: Any):
    figure = plt.figure(figsize=(10, 6), dpi=400)
    axes = figure.add_subplot(111)
    axes.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
