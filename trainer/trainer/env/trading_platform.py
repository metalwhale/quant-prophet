import datetime
import logging
from collections import defaultdict
from enum import IntEnum
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import DQN

from ..asset.base import DailyAsset, DailyPrice
from .asset_pool import AssetPool, calc_polarity_diff


MONTHLY_TRADABLE_DAYS_NUM = 20
YEARLY_TRADABLE_DAYS_NUM = 250


class PositionType(IntEnum):
    BUY = 0
    SELL = 1
    SIDELINE = 2


class Position:
    _date: datetime.date
    _position_type: PositionType
    _entry_price: float
    _amount: float

    def __init__(
        self,
        date: datetime.date, position_type: PositionType, entry_price: float, amount: float,
    ) -> None:
        self._date = date
        self._position_type = position_type
        self._entry_price = entry_price
        self._amount = amount

    @property
    def date(self) -> datetime.date:
        return self._date

    @property
    def position_type(self) -> PositionType:
        return self._position_type

    @property
    def entry_price(self) -> float:
        return self._entry_price

    @property
    def amount(self) -> float:
        return self._amount


# Doc:
# - https://gymnasium.farama.org/v0.29.0/tutorials/gymnasium_basics/environment_creation/
# - https://stable-baselines3.readthedocs.io/en/v2.3.2/guide/custom_env.html
class TradingPlatform(gym.Env):
    class ExtraInfo:
        action_values: Dict[datetime.date, Tuple[float, float, float]]

        def __init__(self) -> None:
            self.action_values = {}

    metadata = {"render_modes": ["rgb_array"]}

    # Control flags
    is_training: bool
    smoothing_position_net: bool
    favorite_symbols: Optional[List[str]]
    figure_num: str

    # Terminology
    # "Episode" and "step":
    # - "Episode" and "step" are terms usually used in reinforcement learning.
    #   In our case, an "episode" consists of several consecutive orders of positions,
    #   and a "step" simply refers to a tradable day.
    # "Date":
    # - A "date" refers to a "tradable day of the date type", i.e., a day on which we can open an position.
    #   We use "date" as the end date of a date range when retrieving historical data.
    #   Note that, unless otherwise explained, we consider a "date" to have this meaning.
    # - An example of a day that is not a "date" according to our definition is the "published day" of an asset,
    #   and all days before the "first date" of an episode (see the `reset` method for details),
    #   because they are not used as the end date when retrieving historical data or for opening positions,
    #   but rather as serial data used for training the model.
    #
    # Example of an episode for a specific asset:
    #   (Note: `~~~~~~~~*` depicts length of `_historical_days_num`,
    #    with `*` being the date used as the end date when retrieving the historical data)
    #
    #   Asset's "published day"
    #   |
    #   |       |<---------------------------------------- Asset's tradable date range ---------------------------------------->|
    #   |       |<------------------------ Training data ------------------------>|<------------- Evaluation data ------------->|
    #   |       |                                                                 |                                             |
    #   ~~~~~~~~|=================================================================|=============================================| Now
    #   |       |                                                                 |
    #   |                                                                         Last training date
    #   |
    #   |<----->| The first `historical_days_num` days are non-tradable and will be skipped
    #
    #                          |<------------- Episode's date range ------------->|
    #                          |==================================================|
    #                          |                                                  |
    #                          |                          ~~~~~~~~*     ---->     |
    #                          |                                  |
    #                          |                                  The "date" will continuously grow until the last training date
    #                  ~~~~~~~~*
    #                          |
    #                          Episode's "first date", randomly chosen within training data's date range

    _asset_pool: AssetPool
    _historical_days_num: int  # Number of days used for retrieving historical data

    # Hyperparameters for calculating rewards
    # TODO: Reconsider the meaning of the daily fee.
    # Its sole purpose currently seems to be only preventing holding a position too long, causing a loss before earning.
    # Does it still make sense since we are always holding every day?
    _position_holding_daily_fee: float  # Positive ratio
    # TODO: Reconsider the meaning of the opening penalty.
    # I believe that changing the opening penalty affects how often new positions are opened,
    # i.e., increasing the opening penalty means the model may learn to open fewer positions.
    _position_opening_penalty: float  # Positive ratio

    # Hyperparameters for termination and truncation
    _max_balance_loss: Optional[float]  # Positive ratio
    _max_balance_gain: Optional[float]  # Positive ratio
    _max_positions_num: Optional[int]  # Maximum number of positions (greater than 1) allowed in one episode
    _max_steps_num: Optional[int]  # Maximum number of steps allowed in one episode

    # State components
    # Platform-level, only changed if we refresh
    _polarity_diff: int
    _date_chosen_counter: Dict[datetime.date, Dict[str, int]]  # Metadata
    # Episode-level, only changed if we reset to begin a new episode
    _asset_symbol: str  # Symbol of the current asset
    _date_range: List[datetime.date]  # The random date range of each episode
    # Step-level, likely changed at each step
    _date_index: int  # Grows in the same episode, resets to 0 for a new episode
    _prices: List[DailyPrice]  # Updated whenever the date changes
    _positions: List[Position]  # Keeps adding positions in the same episode, clears them all for a new episode
    _balance: float  # Resets to initial balance after each episode

    # Extra information
    _extra_info: ExtraInfo

    # Constants, mainly used only for training
    # NOTE: In reality, initial balance should be higher than position amount to cover opening penalty.
    # Here, all set to 1 for simplicity.
    _POSITION_AMOUNT_UNIT: float = 1.0  # Equal to or less than the initial balance
    _INITIAL_BALANCE_UNIT: float = 1.0
    _ASSET_TYPE_WEIGHTS: Tuple[float, float] = [0.0, 1.0]  # (primary, secondary)
    _CLOSE_RANDOM_RADIUS: Optional[int] = 0

    def __init__(
        self,
        asset_pool: AssetPool,
        historical_days_num: int,
        position_holding_daily_fee: float = 0.0,
        position_opening_penalty: float = 0.0,
        max_balance_loss: Optional[float] = None,
        max_balance_gain: Optional[float] = None,
        max_positions_num: Optional[int] = None,
        max_steps_num: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Control flags
        self.is_training = True
        self.smoothing_position_net = False
        self.favorite_symbols = None
        self.figure_num = ""
        # Hyperparameters
        self._asset_pool = asset_pool
        self._historical_days_num = historical_days_num
        # These following hyperparameters are mainly used only for training,
        # by calculating reward and determining whether to terminate an episode.
        self._position_holding_daily_fee = position_holding_daily_fee
        self._position_opening_penalty = position_opening_penalty
        self._max_balance_loss = max_balance_loss
        self._max_balance_gain = max_balance_gain
        self._max_positions_num = max_positions_num
        self._max_steps_num = max_steps_num
        # Environment
        # LINK: Ignore the last position type (SIDELINE), use only BUY and SELL
        self.action_space = gym.spaces.Discrete(len(PositionType) - 1)
        # NOTE: Theoretically, we only need the historical price when deciding the position order (buy/sell or hold).
        # However, with DQN (or perhaps many other RL algorithms),
        # to inform the agent about how long the episode has elapsed and how long it takes to finish the episode,
        # knowing "where" the current state is relatively located in an episode is critically required.
        # In my best guess, this can be solved by adding information about current balance or the position we are holding.
        self.observation_space = gym.spaces.Dict({
            # Suppose that delta values (ratios) are greater than -1 and less than 1,
            # meaning prices and other indicators never drop to 0 and never double from previous day.
            "historical_ema_diffs": gym.spaces.Box(-1, 1, shape=(self._historical_days_num,)),
            # Position types have the same values as action space.
            "position_type": gym.spaces.Discrete(len(PositionType)),
        })
        # Refresh platform-level state components
        self.refresh()

    def reset(
        self, *,
        seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        preferring_secondary = (
            self.is_training and self._asset_pool.is_secondary_generatable
            and np.random.choice([False, True], p=self._ASSET_TYPE_WEIGHTS)
        )
        if preferring_secondary:
            self._asset_pool.renew_secondary_assets()
        self._asset_symbol, self._date_range = self._asset_pool.choose_asset_date(
            favorite_symbols=self.favorite_symbols,
            randomizing_start=self.is_training,
            preferring_secondary=preferring_secondary,
        )
        self._asset.prepare_indicators(close_random_radius=self._CLOSE_RANDOM_RADIUS if self.is_training else None)
        self._date_index = 0
        self._retrieve_prices()
        self._positions = []
        self._balance = self._initial_balance
        observation = self._obtain_observation()
        info = {}
        if self.is_training:
            self._date_chosen_counter[self._date_range[self._date_index]][self._asset_symbol] += 1
            self._polarity_diff += calc_polarity_diff(self._prices[-1].price_delta)
        self._extra_info = self.ExtraInfo()
        return observation, info

    def step(self, action: np.int64) -> tuple[Dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        reward = 0
        # If the position type changes, close the current position and open a new one
        if len(self._positions) == 0 or action != int(self._positions[-1].position_type):
            self._positions.append(Position(
                self._prices[-1].date,
                PositionType(action),
                self._prices[-1].actual_price, self._position_amount,
            ))
            if action != int(PositionType.SIDELINE):
                reward += self._positions[-1].amount * -self._position_opening_penalty  # Opening penalty
        # Recalculate position's net by first reverting net of the current date.
        # The net of the next date will be calculated later.
        reward -= self._positions[-1].amount * self._last_position_net_ratio
        # Move to the next date
        self._date_index += 1
        self._retrieve_prices()
        # TODO: Consider if it is okay to include net of the next date in the reward
        reward += self._positions[-1].amount * self._last_position_net_ratio  # Net of the next date
        if self._positions[-1].position_type != PositionType.SIDELINE:
            reward += self._positions[-1].amount * -self._position_holding_daily_fee  # Holding fee
        # Treat the balance as a cumulative reward in each episode
        self._balance += reward
        # Read more about termination and truncation at:
        # - https://gymnasium.farama.org/v0.29.0/tutorials/gymnasium_basics/handling_time_limits/
        # - https://farama.org/Gymnasium-Terminated-Truncated-Step-API
        # Termination conditions
        terminated = False
        # Truncation conditions
        is_end_of_date = self._date_index >= len(self._date_range) - 1
        truncated = (
            # Reaching the end of training date
            is_end_of_date
            # Liquidated
            or (
                self._max_balance_loss is not None
                and self._balance < self._initial_balance * (1 - self._max_balance_loss)
            )
            # Realizing profits
            or (
                self._max_balance_gain is not None
                and self._balance >= self._initial_balance * (1 + self._max_balance_gain)
            )
            # Other conditions
            or (self._max_positions_num is not None and len(self._positions) >= self._max_positions_num)
            or (self._max_steps_num is not None and self._date_index >= self._max_steps_num)
        )
        # Observation and additional info
        observation = self._obtain_observation()
        info = {
            "is_end_of_date": is_end_of_date,
        }
        if self.is_training:
            self._polarity_diff += calc_polarity_diff(self._prices[-1].price_delta)
        return observation, reward, terminated, truncated, info

    def render(self) -> Any | List[Any] | None:
        YEAR_WIDTH = 2
        SUBPLOT_HEIGHT = 3
        if self.render_mode != "rgb_array":
            return
        # Retrieve prices
        prices = self._asset.retrieve_historical_prices(
            self._date_range[self._date_index],
            # Retrieve all the days before the current date in the date range
            self._date_index + self._historical_days_num,
            randomizing_end=self.is_training,
        )
        dates = [p.date for p in prices]
        plt.rcParams.update({"font.size": 8})
        figure = plt.figure(
            # LINK: `num` and `clear` help prevent memory leak (See: https://stackoverflow.com/a/65910539)
            num=f"trading_platform.{self.figure_num}",
            # NOTE: Remember to increase the height of `figsize` if you add more plots
            figsize=(len(dates) / YEARLY_TRADABLE_DAYS_NUM * YEAR_WIDTH, 4 * SUBPLOT_HEIGHT),
            dpi=400,
            clear=True,
        )
        figure.subplots_adjust(left=100 / len(dates), bottom=0.1, right=0.99, top=0.9)
        all_axes: List[Tuple[Axes, Dict[datetime.date, float]]] = []
        # Plot prices
        axes = figure.add_subplot(411)
        all_axes.append((axes, {p.date: p.actual_price for p in prices}))
        position_index: Optional[int] = None
        date_prices: List[Tuple[datetime.date, float]] = []
        for i, price in enumerate(prices):
            next_position_index = 0 if position_index is None else position_index + 1
            date_price = (price.date, price.actual_price)  # Use actual price and ignore position's entry price
            date_prices.append(date_price)
            if (
                # Position type changes
                (len(self._positions) > next_position_index and price.date == self._positions[next_position_index].date)
                # Last price
                or (i == len(prices) - 1)
            ):
                # Plot dates and prices for current position
                if len(date_prices) > 0:
                    position_type: PositionType
                    if position_index is None:
                        # The days before first position
                        position_type = PositionType.SIDELINE
                    else:
                        position_type = self._positions[position_index].position_type
                    color: str
                    if position_type == PositionType.SIDELINE:
                        color = "gray"
                    elif position_type == PositionType.BUY:
                        color = "green"
                    elif position_type == PositionType.SELL:
                        color = "red"
                    else:
                        color = "black"  # Placeholder
                    axes.plot([d for d, _ in date_prices], [p for _, p in date_prices], color=color)
                # Move to next position
                position_index = next_position_index
                date_prices = [date_price]
        # Plot smoothed prices
        axes = figure.add_subplot(412)
        all_axes.append((axes, {p.date: p.smoothed_price for p in prices}))
        axes.plot(dates, [p.smoothed_price if p.date in dates else None for p in prices])
        # Plot EMA diffs
        axes = figure.add_subplot(413)
        all_axes.append((axes, {p.date: p.ema_diff for p in prices}))
        axes.plot(dates, [0 for _ in dates], color="gray")
        axes.plot(dates, [p.ema_diff if p.date in dates else None for p in prices], color="orange")
        # Plot action values
        action_values = self._extra_info.action_values
        axes = figure.add_subplot(414)
        all_axes.append((axes, {d: v[2] for d, v in action_values.items()}))
        axes.plot(dates, [0 for _ in dates], color="gray")
        axes.plot(dates, [action_values[d][0] if d in action_values else None for d in dates], color="green", alpha=0.2)
        axes.plot(dates, [action_values[d][1] if d in action_values else None for d in dates], color="red", alpha=0.2)
        axes.plot(dates, [action_values[d][2] if d in action_values else None for d in dates], color="orange")
        # Common plots for all axes
        for axes, position_values in all_axes:
            # Plot positions
            for position in self._positions:
                is_buy = position.position_type == PositionType.BUY
                axes.plot(
                    position.date, position_values[position.date],
                    color="green" if is_buy else "red", marker="o", markersize=0.5,
                )
            # Adjust other properties
            for line in axes.lines:
                line.set_linewidth(0.5)
        # Draw figure
        figure.canvas.draw()
        image = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8) \
            .reshape(figure.canvas.get_width_height()[::-1] + (3,))
        return image

    def refresh(self):
        self._polarity_diff = 0
        self._date_chosen_counter = defaultdict(lambda: defaultdict(int))

    def trade(
        self,
        model: Optional[BaseAlgorithm] = None,
        action_diff_threshold: float = 0.0,
        max_step: Optional[int] = None,
        stopping_when_done: bool = True,
        rendering: bool = True,
    ) -> Tuple[Any, Tuple[float, ...], Tuple[List[Position], DailyPrice]]:
        self.render_mode = "rgb_array"
        obs, _ = self.reset()
        # Run one episode
        step = 0
        action = None
        while max_step is None or step < max_step:
            if model is not None:
                # See:
                # - https://github.com/DLR-RM/stable-baselines3/blob/v2.3.2/stable_baselines3/common/base_class.py#L536
                # - https://github.com/DLR-RM/stable-baselines3/blob/v2.3.2/stable_baselines3/common/policies.py#L331
                model.policy.set_training_mode(False)
                obs_tensor, _ = model.policy.obs_to_tensor(obs)
                action_values: torch.Tensor
                with torch.no_grad():
                    if isinstance(model, DQN):
                        # See:
                        # - https://github.com/DLR-RM/stable-baselines3/blob/v2.3.2/stable_baselines3/dqn/policies.py#L183
                        # - https://github.com/DLR-RM/stable-baselines3/blob/v2.3.2/stable_baselines3/dqn/policies.py#L68
                        action_values = model.policy.q_net(obs_tensor)
                    else:
                        # TODO: Handle other algorithms
                        raise NotImplementedError
                # LINK: Ignore the last position type (SIDELINE), use only BUY and SELL
                buy_value, sell_value = action_values.cpu().numpy().squeeze()
                action_diff = buy_value - sell_value
                self._extra_info.action_values[self._prices[-1].date] = (buy_value, sell_value, action_diff)
                if abs(action_diff) >= action_diff_threshold or action is None:
                    action = PositionType.BUY if action_diff >= 0 else PositionType.SELL
            else:
                action = int(np.random.choice([PositionType.SIDELINE, PositionType.BUY, PositionType.SELL]))
            obs, _, terminated, truncated, info = self.step(action)
            is_end_of_date = info["is_end_of_date"]
            logging.debug("%s %f %f", self._prices[-1].date, self._prices[-1].actual_price, self._balance)
            if is_end_of_date or (stopping_when_done and (terminated or truncated)):
                break
            step += 1
        rendered = self.render() if rendering else None
        # Calculate the balance
        calculated_balance = self._initial_balance
        earning, price_change, wl_rate = calc_earning(
            self._positions, self._prices[-1],
            # TODO: Include fees even if not in training mode
            position_holding_daily_fee=self._position_holding_daily_fee if self.is_training else 0,
            position_opening_penalty=self._position_opening_penalty if self.is_training else 0,
        )
        calculated_balance += earning
        logging.debug("%s %f", self._prices[-1].date, self._prices[-1].actual_price)
        platform_balance = self._balance
        # Platform balance and self-calculated balance should be equal
        return (
            rendered,
            (platform_balance, calculated_balance, price_change, wl_rate),
            (self._positions, self._prices[-1]),
        )

    @property
    def asset_pool(self) -> AssetPool:
        return self._asset_pool

    @property
    def _asset(self) -> DailyAsset:
        return self._asset_pool.get_asset(self._asset_symbol)

    @property
    def _position_amount(self) -> float:
        # Unit price is mainly used for training, whereas using actual price as a position amount often implies evaluation
        return self._POSITION_AMOUNT_UNIT if self.is_training else self._prices[-1].actual_price

    @property
    def _initial_balance(self) -> float:
        # A zero initial balance may seem illogical, but in evaluation, only earnings matter, not the initial balance
        return self._INITIAL_BALANCE_UNIT if self.is_training else 0

    @property
    def _last_position_net_ratio(self) -> float:
        return calc_position_net_ratio(
            self._positions[-1],
            self._prices[-1].smoothed_price if self.smoothing_position_net else self._prices[-1].actual_price,
        )

    # Should be called right after updating `self._date_index` to the newest date
    def _retrieve_prices(self):
        # Since `retrieve_historical_prices` chooses the end price from a random time on the same end date,
        # multiple calls produce different results for the end price.
        # We need to avoid calling it again if the current date matches when we previously retrieved prices.
        date = self._date_range[self._date_index]
        if hasattr(self, "_prices") and len(self._prices) > 0 and self._prices[-1].date == date:
            return
        self._prices = self._asset.retrieve_historical_prices(
            date, self._historical_days_num,
            randomizing_end=self.is_training,
        )

    def _obtain_observation(self) -> Dict[str, Any]:
        # See: https://stackoverflow.com/questions/73922332/dict-observation-space-for-stable-baselines3-not-working
        return {
            "historical_ema_diffs": np.array(self._magnitude_scale([p.ema_diff for p in self._prices], 1)),
            "position_type": np.array([
                self._positions[-1].position_type if len(self._positions) > 0 else PositionType(
                    # LINK: Ignore the last position type (SIDELINE), use only BUY and SELL
                    self.np_random.choice([PositionType.BUY, PositionType.SELL])
                ),
            ], dtype=int),
        }

    @staticmethod
    def _magnitude_scale(array: List[float], factor: float) -> List[float]:
        return [v * factor for v in array]


def calc_position_net_ratio(position: Position, price: float) -> float:
    position_type = position.position_type
    if position_type == PositionType.SIDELINE:
        return 0
    else:
        return (price / position.entry_price - 1) * (1 if position_type == PositionType.BUY else -1)


def calc_earning(
    positions: List[Position], final_price: DailyPrice,
    position_holding_daily_fee: float = 0.0, position_opening_penalty: float = 0.0,
) -> Tuple[float, float, float]:
    if len(positions) < 1 or positions[-1].date > final_price.date:
        return (0, 0, 0)
    earning = 0
    position_net_ratios: List[float] = []
    price_change_ratios: List[float] = []
    # Rewards (net, fee, penalty) of closed positions (excluding the last position, as it is probably not yet closed)
    for prev_position, cur_position in zip(positions[:-1], positions[1:]):
        # Position net
        position_net_ratio = calc_position_net_ratio(prev_position, cur_position.entry_price)
        position_net_ratios.append(position_net_ratio)
        price_change_ratios.append(cur_position.entry_price / prev_position.entry_price - 1)
        earning += prev_position.amount * position_net_ratio
        if prev_position.position_type != PositionType.SIDELINE:
            # Holding fee
            earning += (cur_position.date - prev_position.date).days \
                * prev_position.amount * -position_holding_daily_fee
            # TODO: Skip calculating penalties when evaluating
            # Opening penalty
            earning += prev_position.amount * -position_opening_penalty
        logging.debug("%s %f %s", prev_position.date, prev_position.entry_price, prev_position.position_type)
    # Reward of the last position
    last_position_net_ratio = calc_position_net_ratio(positions[-1], final_price.actual_price)
    position_net_ratios.append(last_position_net_ratio)
    price_change_ratios.append(final_price.actual_price / positions[-1].entry_price - 1)
    earning += positions[-1].amount * last_position_net_ratio
    if positions[-1].position_type != PositionType.SIDELINE:
        earning += (final_price.date - positions[-1].date).days * positions[-1].amount * -position_holding_daily_fee
        earning += positions[-1].amount * -position_opening_penalty
    logging.debug("%s %f %s", positions[-1].date, positions[-1].entry_price, positions[-1].position_type)
    # Price change equals a BUY position, hence `1` instead of `-1`
    price_change = positions[0].amount * 1 * (final_price.actual_price / positions[0].entry_price - 1)
    # Win-lose rate
    wl_rate = sum([
        # Position net ratios and price change ratios always have the same absolute value but differ in sign,
        # depending on the position type and whether the price is going up or down.
        1 if pnr > pcr else -1 if pnr < pcr else 0
        # Use `pnr if pnr != pcr else 0` to calculate average net ratio
        for pnr, pcr in zip(position_net_ratios, price_change_ratios)
    ]) / len(position_net_ratios)
    return earning, price_change, wl_rate
