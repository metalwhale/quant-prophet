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
    _showing_image: bool

    _ep_count: int
    _general_overview_results: List[Dict[str, Any]]
    _asset_overview_results: Dict[str, List[Dict[str, Any]]]

    _OVERVIEW_LOG_FILE_NAME = "overview.csv"
    _OVERVIEW_CHART_FILE_NAME = "overview.png"

    def __init__(
        self,
        output_path: Path, envs: Dict[str, TradingPlatform], freq: int,
        showing_image: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        # Parameters
        self._output_path = output_path
        self._envs = envs
        self._freq = freq
        self._showing_image = showing_image
        # Initialization
        os.makedirs(output_path, exist_ok=True)
        self._ep_count = 0
        self._general_overview_results = []
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
        general_result: Dict[str, Any] = {"ep_count": self._ep_count}
        asset_results: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"ep_count": self._ep_count})
        for env_name, env in self._envs.items():
            env.set_mode(False)  # Just in case
            earning_ratio_sum = 0.0
            win_count = 0
            for symbol in env.asset_pool.symbols:
                os.makedirs(self._output_path / symbol, exist_ok=True)
                env.favorite_symbols = [symbol]
                (
                    rendered,
                    (_, earning, price_change, wl_ratio),
                    (positions, last_price),
                ) = env.trade(model=self.model, stopping_when_done=False)
                initial_amount = positions[0].amount
                earning_ratio = (initial_amount + earning) / (initial_amount + price_change) - 1
                asset_results[symbol] |= {
                    "ep_count": self._ep_count,
                    f"{env_name}_earning_ratio": earning_ratio,
                    f"{env_name}_wl_ratio": wl_ratio,
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
                    f"wl_ratio={wl_ratio:.2f}",
                ]), fill=(0, 0, 0), font_size=80)
                if self._showing_image:
                    plt.close("all")
                    show_image(image)
                image.save(self._output_path / symbol / f"trade_{env_name}_{self._ep_count}.png")
                # For general overview
                earning_ratio_sum += earning_ratio
                win_count += 1 if earning_ratio >= 0 else 0
            general_result |= {
                f"{env_name}_earning_ratio": earning_ratio_sum / len(env.asset_pool.symbols),
                f"{env_name}_win_rate": win_count / len(env.asset_pool.symbols),
            }
            env.favorite_symbols = None
        # Save overview results
        for symbol in set([s for e in self._envs.values() for s in e.asset_pool.symbols]):
            self.__save_overview_results(symbol, asset_results[symbol])
        self.__save_overview_results(None, general_result)

    def __save_overview_results(self, asset_symbol: Optional[str], result: Dict[str, Any]):
        overview_results: List[Dict[str, Any]]
        overview_log_file_path: Path
        overview_chart_file_path: Path
        wl_field_name: str
        if asset_symbol is None:  # General
            overview_results = self._general_overview_results
            overview_log_file_path = self._output_path / self._OVERVIEW_LOG_FILE_NAME
            overview_chart_file_path = self._output_path / self._OVERVIEW_CHART_FILE_NAME
            wl_field_name = "win_rate"
        else:
            overview_results = self._asset_overview_results[asset_symbol]
            overview_log_file_path = self._output_path / asset_symbol / self._OVERVIEW_LOG_FILE_NAME
            overview_chart_file_path = self._output_path / asset_symbol / self._OVERVIEW_CHART_FILE_NAME
            wl_field_name = "wl_ratio"
        # Write overview log
        overview_log_field_names = [
            "ep_count",
            *[f for n in self._envs.keys() for f in [f"{n}_earning_ratio", f"{n}_{wl_field_name}"]],
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
            axes.plot(ep_counts, [r[f"{env_name}_earning_ratio"] for r in overview_results], color="blue")
            axes.plot(ep_counts, [r[f"{env_name}_{wl_field_name}"] for r in overview_results], color="orange")
        figure.canvas.draw()
        image = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8) \
            .reshape(figure.canvas.get_width_height()[::-1] + (3,))
        if self._showing_image:
            plt.close("all")
        Image.fromarray(image).save(overview_chart_file_path)


def show_image(image: Any):
    figure = plt.figure(figsize=(10, 6), dpi=400)
    axes = figure.add_subplot(111)
    axes.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
