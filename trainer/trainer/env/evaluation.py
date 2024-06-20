import logging
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from .trading_platform import PositionType, TradingPlatform, calc_earning


class FullEvalCallback(BaseCallback):
    _eval_freq: int
    _val_env: TradingPlatform
    _test_env: Optional[TradingPlatform]
    _ep_count: int

    def __init__(
        self,
        eval_freq: int, val_env: TradingPlatform, test_env: Optional[TradingPlatform] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._eval_freq = eval_freq
        self._val_env = val_env
        self._test_env = test_env
        self._ep_count = 0

    def _on_training_start(self) -> None:
        self.__eval_model()
        return super()._on_training_start()

    def _on_step(self) -> bool:
        # See: https://github.com/DLR-RM/stable-baselines3/blob/v2.3.2/stable_baselines3/common/callbacks.py#L590-L631
        if np.sum(self.locals["dones"]).item() != 0:
            self._ep_count += 1
            if self._ep_count > 0 and self._ep_count % self._eval_freq == 0:
                self.__eval_model()
        return super()._on_step()

    def _on_training_end(self) -> None:
        self.__eval_model()
        return super()._on_training_end()

    def __eval_model(self):
        eval_model(self.model, self._val_env, test_env=self._test_env)


def trade(
    env: TradingPlatform,
    model: Optional[BaseAlgorithm] = None, max_step: Optional[int] = None, stop_when_done: bool = True,
    render: bool = True,
) -> Tuple[Any, Tuple[float, ...]]:
    env.render_mode = "rgb_array"
    obs, _ = env.reset()
    # TODO: Avoid using private attributes
    # Run one episode
    step = 0
    while max_step is None or step < max_step:
        action = None
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = int(np.random.choice([PositionType.LONG, PositionType.SHORT]))
        obs, _, terminated, truncated, info = env.step(action)
        is_end_of_date = info["is_end_of_date"]
        logging.debug("%s %f %f", env._prices[-1].date, env._prices[-1].actual_price, env._balance)
        if is_end_of_date or (stop_when_done and (terminated or truncated)):
            break
        step += 1
    rendered = env.render() if render else None
    # Calculate the balance
    self_calculated_balance = env._initial_balance
    earning, actual_price_change = calc_earning(
        env._positions, env._prices[-1],
        position_opening_fee=env._position_opening_fee if env.is_training_mode else 0,
    )
    self_calculated_balance += earning
    logging.debug("%s %f", env._prices[-1].date, env._prices[-1].actual_price)
    platform_balance = env._balance
    # Platform balance and self-calculated balance should be equal
    return rendered, (platform_balance, self_calculated_balance, actual_price_change)


def eval_model(model: BaseAlgorithm, val_env: TradingPlatform, test_env: Optional[TradingPlatform] = None):
    rendered, (_, earning, actual_price_change) = trade(val_env, model=model, stop_when_done=False)
    show_image(rendered, text=f"earning={earning:.2f}, actual_price_change={actual_price_change:.2f}")
    if test_env is not None:
        rendered, (_, earning, actual_price_change) = trade(test_env, model=model, stop_when_done=False)
        show_image(rendered, text=f"earning={earning:.2f}, actual_price_change={actual_price_change:.2f}")


def show_image(image: Any, text: str = ""):
    figure = plt.figure(figsize=(10, 6), dpi=800)
    axes = figure.add_subplot(111)
    axes.imshow(image)
    if text != "":
        axes.text(0, 0, text)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
