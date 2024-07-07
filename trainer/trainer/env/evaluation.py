import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from ..asset.base import DailyPrice
from .trading_platform import Position, PositionType, TradingPlatform, calc_earning


class FullEvalCallback(BaseCallback):
    _output_path: Path
    _envs: Dict[str, TradingPlatform]
    _freq: int
    _action_diff_threshold: Optional[float]
    _showing_image: bool

    _ep_count: int
    _overview_records: List[Dict[str, Any]]

    _OVERVIEW_LOG_FILE_NAME = "overview.csv"
    _OVERVIEW_CHART_FILE_NAME = "overview.png"

    def __init__(
        self,
        output_path: Path, envs: Dict[str, TradingPlatform], freq: int,
        action_diff_threshold: Optional[int] = None, showing_image: bool = True,
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
        self._overview_records = []

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
        row: Dict[str, Any] = {"ep_count": self._ep_count}
        for env_name, env in self._envs.items():
            env.is_training = False  # Just in case
            # TODO: Iterate through all assets
            (
                rendered,
                (_, earning, price_change),
                (positions, last_price),
            ) = trade(
                env,
                model=self.model,
                action_diff_threshold=self._action_diff_threshold, stopping_when_done=False,
            )
            # Write trade positions
            with open(self._output_path / f"trade_{self._ep_count}_{env_name}.csv", "w") as positions_file:
                positions_writer = csv.DictWriter(positions_file, ["date", "entry_price", "position_type"])
                for position in positions:
                    positions_writer.writerow({
                        "date": position.date,
                        "entry_price": position.entry_price,
                        "position_type": position.position_type.name,
                    })
                positions_writer.writerow({
                    "date": last_price.date,
                    "entry_price": last_price.actual_price,
                })
            # Draw trade chart
            image = Image.fromarray(rendered)
            draw = ImageDraw.Draw(image)
            draw.text((60, 60), f"Episode {self._ep_count}: " + ", ".join([
                f"{env_name}_earning={earning:.2f}",
                f"{env_name}_price_change={price_change:.2f}",
            ]), fill=(0, 0, 0), font_size=120)
            if self._showing_image:
                plt.close("all")
                show_image(image)
            image.save(self._output_path / f"trade_{self._ep_count}_{env_name}.png")
            row |= {
                f"{env_name}_earning": earning,
                f"{env_name}_price_change": price_change,
                f"{env_name}_earning_discrepancy": earning / price_change - 1,
            }
        # Write overview log
        overview_log_field_names = [
            "ep_count",
            *[f for n in self._envs.keys() for f in [f"{n}_earning", f"{n}_earning_discrepancy"]],
        ]
        overview_log_file_path = self._output_path / self._OVERVIEW_LOG_FILE_NAME
        if not os.path.isfile(overview_log_file_path):
            with open(self._output_path / self._OVERVIEW_LOG_FILE_NAME, "w") as overview_log_file:
                overview_log_writer = csv.DictWriter(overview_log_file, overview_log_field_names)
                overview_log_writer.writeheader()
                overview_log_file.flush()
        with open(overview_log_file_path, "a") as overview_log_file:
            overview_log_writer = csv.DictWriter(overview_log_file, overview_log_field_names)
            overview_log_writer.writerow({k: v for k, v in row.items() if k in overview_log_field_names})
            overview_log_file.flush()
        # Draw overview chart
        self._overview_records.append(row)
        # LINK: `num` and `clear` help prevent memory leak (See: https://stackoverflow.com/a/65910539)
        # `num=2` is reserved for overview chart
        figure = plt.figure(figsize=(10, 3 * len(self._envs)), dpi=800, num=2, clear=True)
        figure.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9)
        for i, env_name in enumerate(self._envs.keys()):
            axes = figure.add_subplot(len(self._envs), 1, i + 1)
            axes.plot(
                [r["ep_count"] for r in self._overview_records],
                [0 for _ in self._overview_records],
                color="gray",
            )
            axes.plot(
                [r["ep_count"] for r in self._overview_records],
                [r[f"{env_name}_earning_discrepancy"] for r in self._overview_records],
                color="blue",
            )
        figure.canvas.draw()
        image = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8) \
            .reshape(figure.canvas.get_width_height()[::-1] + (3,))
        if self._showing_image:
            plt.close("all")
        Image.fromarray(image).save(self._output_path / self._OVERVIEW_CHART_FILE_NAME)


def trade(
    env: TradingPlatform,
    model: Optional[BaseAlgorithm] = None,
    action_diff_threshold: Optional[float] = None,
    max_step: Optional[int] = None,
    stopping_when_done: bool = True,
    rendering: bool = True,
) -> Tuple[Any, Tuple[float, ...], Tuple[List[Position], DailyPrice]]:
    env.render_mode = "rgb_array"
    obs, _ = env.reset()
    # TODO: Avoid using private attributes
    # Run one episode
    step = 0
    action = None
    while max_step is None or step < max_step:
        if model is not None:
            if action_diff_threshold is None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                model.policy.set_training_mode(False)
                obs_tensor, _ = model.policy.obs_to_tensor(obs)
                with torch.no_grad():
                    q_values = model.policy.q_net(obs_tensor)
                # LINK: Ignore the last position type (SIDELINE), use only BUY and SELL
                v1, v2 = q_values.numpy()[0]
                if abs(v1 / v2 - 1) >= action_diff_threshold or action is None:
                    action = 0 if v1 > v2 else 1
        else:
            action = int(np.random.choice([PositionType.SIDELINE, PositionType.BUY, PositionType.SELL]))
        obs, _, terminated, truncated, info = env.step(action)
        is_end_of_date = info["is_end_of_date"]
        logging.debug("%s %f %f", env._prices[-1].date, env._prices[-1].actual_price, env._balance)
        if is_end_of_date or (stopping_when_done and (terminated or truncated)):
            break
        step += 1
    rendered = env.render() if rendering else None
    # Calculate the balance
    calculated_balance = env._initial_balance
    earning, price_change = calc_earning(
        env._positions, env._prices[-1],
        # TODO: Include fees even if not in training mode
        position_holding_daily_fee=env._position_holding_daily_fee if env.is_training else 0,
        position_opening_penalty=env._position_opening_penalty if env.is_training else 0,
        short_period_penalty=env._short_period_penalty if env.is_training else 0,
    )
    calculated_balance += earning
    logging.debug("%s %f", env._prices[-1].date, env._prices[-1].actual_price)
    platform_balance = env._balance
    # Platform balance and self-calculated balance should be equal
    return (
        rendered,
        (platform_balance, calculated_balance, price_change),
        (env._positions, env._prices[-1]),
    )


def show_image(image: Any):
    figure = plt.figure(figsize=(10, 6), dpi=800)
    axes = figure.add_subplot(111)
    axes.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
