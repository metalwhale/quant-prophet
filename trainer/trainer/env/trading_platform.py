import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import DQN

from ..asset.base import DailyAsset, DailyPrice, LevelType
from .asset_pool import AssetPool


MONTHLY_TRADABLE_DAYS_NUM = 20
YEARLY_TRADABLE_DAYS_NUM = 250


class PositionType(Enum):
    _sign: int

    BUY = 0, 1
    SELL = 1, -1
    SIDELINE = 2, 0

    # LINK: See: https://stackoverflow.com/a/54732120
    def __new__(cls, *args) -> "PositionType":
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: int, sign: int) -> None:
        self._sign = sign

    @property
    def sign(self) -> int:
        return self._sign


class Position:
    _date: datetime.date
    _position_type: PositionType
    _entry_price: DailyPrice
    _amount: float

    def __init__(
        self,
        date: datetime.date, position_type: PositionType, entry_price: DailyPrice, amount: float,
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
    def entry_price(self) -> DailyPrice:
        return self._entry_price

    @property
    def amount(self) -> float:
        return self._amount


class PriceType(Enum):
    ACTUAL = 0
    MODIFIED = 1


class AmountType(Enum):
    UNIT = 0
    SPOT = 1


# Doc:
# - https://gymnasium.farama.org/v0.29.0/tutorials/gymnasium_basics/environment_creation/
# - https://stable-baselines3.readthedocs.io/en/v2.3.2/guide/custom_env.html
class TradingPlatform(gym.Env):
    class ExtraInfo:
        action_values: Dict[datetime.date, Tuple[float, float, float]]
        earning: float

        def __init__(self) -> None:
            self.action_values = {}
            self.earning = 0

    metadata = {"render_modes": ["rgb_array"]}

    favorite_symbols: Optional[List[str]]
    figure_num: str

    # Control flags
    _randomizing: bool
    _position_net_price_type: PriceType
    _position_amount_type: AmountType

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

    # Parameters
    _asset_pool: AssetPool
    _historical_days_num: int  # Number of days used for retrieving historical data

    # Hyperparameters
    _position_opening_penalty: float = 0.0

    # State components
    # Episode-level, only changed if we reset to begin a new episode
    _asset_symbol: str  # Symbol of the current asset
    _date_range: List[datetime.date]  # The random date range of each episode
    # Step-level, likely changed at each step
    _date_index: int  # Grows in the same episode, resets to 0 for a new episode
    _prices: List[DailyPrice]  # Updated whenever the date changes
    _positions: List[Position]  # Keeps adding positions in the same episode, clears them all for a new episode

    # Extra information
    _extra_info: ExtraInfo

    # Constants, mainly used only for training
    _POSITION_AMOUNT_UNIT: float = 100.0
    _RANDOM_RADIUS: Optional[int] = 0

    def __init__(
        self,
        asset_pool: AssetPool,
        historical_days_num: int,
    ) -> None:
        super().__init__()
        self.favorite_symbols = None
        self.figure_num = ""
        # Parameters
        self._asset_pool = asset_pool
        self._historical_days_num = historical_days_num
        # Environment
        # LINK: Ignore the last position type (SIDELINE), use only BUY and SELL
        self.action_space = gym.spaces.Discrete(len(PositionType) - 1)
        self.observation_space = gym.spaces.Dict({
            "historical_price_delta_ratios": gym.spaces.Box(-1, 1, shape=(self._historical_days_num,)),
            "historical_ema_diff_ratios": gym.spaces.Box(-1, 1, shape=(self._historical_days_num,)),
            "historical_scaled_rsis": gym.spaces.Box(0, 1, shape=(self._historical_days_num,)),
            "historical_scaled_adxs": gym.spaces.Box(0, 1, shape=(self._historical_days_num,)),
            "historical_scaled_ccis": gym.spaces.Box(-1, 1, shape=(self._historical_days_num,)),
            "last_position_type": gym.spaces.Discrete(len(PositionType)),
            "last_position_net_ratio": gym.spaces.Box(-1, 1, shape=(1,)),
        })

    def reset(
        self, *,
        seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._asset_symbol, self._date_range = self._asset_pool.choose_asset_date(
            favorite_symbols=self.favorite_symbols,
            randomizing_start=self._randomizing,
        )
        self._asset.prepare_indicators(random_radius=self._RANDOM_RADIUS if self._randomizing else None)
        self._date_index = 0
        self._retrieve_prices()
        self._positions = []
        observation = self._obtain_observation()
        info = {}
        self._extra_info = self.ExtraInfo()
        return observation, info

    def step(self, action: np.int64) -> tuple[Dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        reward = 0
        # If the position type changes, close the current position and open a new one
        if len(self._positions) == 0 or action != self._positions[-1].position_type.value:
            amount = self._POSITION_AMOUNT_UNIT if self._position_amount_type == AmountType.UNIT \
                else self._prices[-1].actual_price
            position = Position(self._prices[-1].date, PositionType(action), self._prices[-1], amount)
            self._positions.append(position)
            # Penalty for opening a new position
            # TODO: Consider which types of positions we should penalize (currently all)
            reward += position.amount * -self._position_opening_penalty
        # Recalculate position's net by first reverting net of the current date.
        # The net of the next date will be calculated later.
        earning = -self._positions[-1].amount * self._last_position_net_ratio
        # Move to the next date
        self._date_index += 1
        self._retrieve_prices()
        # TODO: Consider if it is okay to include net of the next date in the reward
        earning += self._positions[-1].amount * self._last_position_net_ratio  # Net of the next date
        self._extra_info.earning += earning
        # Use only the earning of SELL positions for rewards because:
        # - BUY positions are similar to holding, so performance depends mainly on SELL strategy.
        # - The optimal position amount for calculating rewards is unclear, whether using a fixed unit or the actual price.
        # Note that this doesn't mean it's not good to include the earning of BUY positions for reward.
        if self._positions[-1].position_type == PositionType.SELL:
            reward += earning
        # Read more about termination and truncation at:
        # - https://gymnasium.farama.org/v0.29.0/tutorials/gymnasium_basics/handling_time_limits/
        # - https://farama.org/Gymnasium-Terminated-Truncated-Step-API
        # Termination conditions
        terminated = False
        # Truncation conditions
        is_end_of_date = self._date_index >= len(self._date_range) - 1
        truncated = is_end_of_date  # Reaching the end of training date
        # Observation and additional info
        observation = self._obtain_observation()
        info = {
            "is_end_of_date": is_end_of_date,
        }
        return observation, reward, terminated, truncated, info

    def render(self) -> Any | List[Any] | None:
        YEAR_WIDTH = 2
        SUBPLOT_HEIGHT = 3
        FEATURES = ["ema_diff_ratio", "scaled_rsi", "scaled_adx", "scaled_cci"]
        TOP_MARGIN = 350
        BOTTOM_MARGIN = 150
        subplots_num = len(FEATURES) + 2  # 2 subplots other than features are for prices and action values
        if self.render_mode != "rgb_array":
            return
        # Retrieve prices
        prices = self._asset.retrieve_historical_prices(
            self._date_range[self._date_index],
            # Retrieve all the days before the current date in the date range
            self._date_index + self._historical_days_num,
        )
        # TODO: Adjust the x-axis to represent dates linearly, rather than based on datetime values,
        # to ensure that the distance between consecutive data points remains consistent.
        dates = [p.date for p in prices]
        plt.rcParams.update({"font.size": 8})
        figure = plt.figure(
            # LINK: `num` and `clear` help prevent memory leak (See: https://stackoverflow.com/a/65910539)
            num=f"trading_platform.{self.figure_num}",
            # NOTE: Remember to increase the height of `figsize` if you add more plots
            figsize=(len(dates) / YEARLY_TRADABLE_DAYS_NUM * YEAR_WIDTH, subplots_num * SUBPLOT_HEIGHT),
            dpi=400,
            clear=True,
        )
        figure_height = (figure.get_size_inches() * figure.dpi)[1]
        vertical_padding = figure_height / subplots_num * 0.00002  # NOTE: Hard-coded, adjust if the figure size changes
        figure.subplots_adjust(
            left=100 / len(dates), right=0.99,
            top=1 - TOP_MARGIN / figure_height, bottom=BOTTOM_MARGIN / figure_height,
        )
        all_axes: List[Tuple[str, Axes, Dict[datetime.date, float]]] = []
        # Plot prices
        axes = figure.add_subplot(subplots_num, 1, 1)
        all_axes.append(("price", axes, {p.date: p.actual_price for p in prices}))
        actual_prices = [p.actual_price for p in prices]
        actual_prices_range = max(actual_prices) - min(actual_prices)
        position_index: Optional[int] = None
        date_prices: List[Tuple[datetime.date, float]] = []
        for i, price in enumerate(prices):
            next_position_index = 0 if position_index is None else position_index + 1
            date_price = (price.date, price.actual_price)  # Use actual price and ignore position's entry price
            date_prices.append(date_price)
            # Plot position price
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
            # Plot level
            if price.level_type is not None:
                is_basic = price.level_type == LevelType.SUPPORT or price.level_type == LevelType.RESISTANCE
                is_support = price.level_type == LevelType.SUPPORT or price.level_type == LevelType.SUPPORT_PULLBACK
                padding = actual_prices_range * vertical_padding / price.actual_price
                axes.plot(
                    price.date, price.actual_price * (1 + (-1 if is_support else 1) * padding),
                    color="green" if is_support else "red",
                    marker="^" if is_support else "v",
                    alpha=1 if is_basic else 0.5, markersize=0.5,
                )
        axes.plot(dates, [p.modified_price if p.date in dates else None for p in prices], alpha=0.5)
        # Plot features
        for i, feature in enumerate(FEATURES):
            is_ratio = "ratio" in feature
            axes = figure.add_subplot(subplots_num, 1, i + 2)
            all_axes.append((feature.replace("_", " "), axes, {p.date: getattr(p, feature) for p in prices}))
            if is_ratio:
                axes.plot(dates, [0 for _ in dates], color="gray")
            axes.plot(
                dates, [getattr(p, feature) if p.date in dates else None for p in prices],
                color="orange" if is_ratio else None,
            )
        # Plot action values
        action_values = self._extra_info.action_values
        axes = figure.add_subplot(subplots_num, 1, subplots_num)
        all_axes.append(("action value", axes, {d: v[2] for d, v in action_values.items()}))
        axes.plot(dates, [0 for _ in dates], color="gray")
        axes.plot(dates, [action_values[d][0] if d in action_values else None for d in dates], color="green", alpha=0.2)
        axes.plot(dates, [action_values[d][1] if d in action_values else None for d in dates], color="red", alpha=0.2)
        axes.plot(dates, [action_values[d][2] if d in action_values else None for d in dates], color="orange")
        # Common plots for all axes
        for title, axes, position_values in all_axes:
            axes.set_title(title)
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
        action: Optional[PositionType] = None
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
                action = np.random.choice([PositionType.SIDELINE, PositionType.BUY, PositionType.SELL])
            obs, _, terminated, truncated, info = self.step(action.value)
            is_end_of_date = info["is_end_of_date"]
            if is_end_of_date or (stopping_when_done and (terminated or truncated)):
                break
            step += 1
        rendered = self.render() if rendering else None
        # Calculate the earning
        calculated_earning, price_change, wl_ratio = _calc_earning(
            self._positions, self._prices[-1],
            self._position_net_price_type,
        )
        # Platform earning and self-calculated earning should be equal
        platform_earning = self._extra_info.earning
        return (
            rendered,
            (platform_earning, calculated_earning, price_change, wl_ratio),
            (self._positions, self._prices[-1]),
        )

    def set_mode(self, is_training: bool):
        self._randomizing = is_training
        self._position_net_price_type = PriceType.MODIFIED if is_training else PriceType.ACTUAL
        self._position_amount_type = AmountType.UNIT if is_training else AmountType.SPOT

    @property
    def asset_pool(self) -> AssetPool:
        return self._asset_pool

    @property
    def _asset(self) -> DailyAsset:
        return self._asset_pool.get_asset(self._asset_symbol)

    @property
    def _last_position_net_ratio(self) -> float:
        return _calc_position_net_ratio(self._positions[-1], self._prices[-1], self._position_net_price_type)

    # Should be called right after updating `self._date_index` to the newest date
    def _retrieve_prices(self):
        # Since `retrieve_historical_prices` chooses the end price from a random time on the same end date,
        # multiple calls produce different results for the end price.
        # We need to avoid calling it again if the current date matches when we previously retrieved prices.
        date = self._date_range[self._date_index]
        if hasattr(self, "_prices") and len(self._prices) > 0 and self._prices[-1].date == date:
            return
        self._prices = self._asset.retrieve_historical_prices(date, self._historical_days_num)

    def _obtain_observation(self) -> Dict[str, Any]:
        # Observation for historical features
        historical_ema_diff_ratios = [p.ema_diff_ratio for p in self._prices]
        historical_scaled_rsis = [p.scaled_rsi for p in self._prices]
        historical_scaled_adxs = [p.scaled_adx for p in self._prices]
        historical_scaled_ccis = [p.scaled_cci for p in self._prices]
        # Observation for the last position
        last_position_type: PositionType = PositionType(
            # LINK: Ignore the last position type (SIDELINE), use only BUY and SELL
            self.np_random.choice([PositionType.BUY, PositionType.SELL])
        )
        last_position_net_ratio: float = 0
        if len(self._positions) > 0:
            last_position_type = self._positions[-1].position_type
            last_position_net_ratio = self._last_position_net_ratio
        # See: https://stackoverflow.com/questions/73922332/dict-observation-space-for-stable-baselines3-not-working
        return {
            "historical_price_delta_ratios": np.array([p.price_delta_ratio for p in self._prices]),
            "historical_ema_diff_ratios": np.array(historical_ema_diff_ratios),
            "historical_scaled_rsis": np.array(historical_scaled_rsis),
            "historical_scaled_adxs": np.array(historical_scaled_adxs),
            "historical_scaled_ccis": np.array(historical_scaled_ccis),
            "last_position_type": np.array([last_position_type.value], dtype=int),
            "last_position_net_ratio": np.array([last_position_net_ratio]),
        }


def _calc_position_net_ratio(position: Position, price: DailyPrice, price_type: PriceType) -> float:
    entry_price: float
    market_price: float
    if price_type == PriceType.ACTUAL:
        entry_price = position.entry_price.actual_price
        market_price = price.actual_price
    elif price_type == PriceType.MODIFIED:
        entry_price = position.entry_price.modified_price
        market_price = price.modified_price
    else:
        raise NotImplementedError
    return position.position_type.sign * (market_price / entry_price - 1)


def _calc_earning(
    positions: List[Position], last_price: DailyPrice,
    price_type: PriceType,
) -> Tuple[float, float, float]:
    if len(positions) < 1 or positions[-1].date > last_price.date:
        return (0, 0, 0)
    earning = 0
    position_net_ratios: List[float] = []
    price_change_ratios: List[float] = []
    # Earning of closed positions (excluding the last position, as it is probably not yet closed)
    for prev_position, cur_position in zip(positions[:-1], positions[1:]):
        # Position net
        position_net_ratio = _calc_position_net_ratio(prev_position, cur_position.entry_price, price_type)
        position_net_ratios.append(position_net_ratio)
        price_change_ratios.append(cur_position.entry_price.actual_price / prev_position.entry_price.actual_price - 1)
        earning += prev_position.amount * position_net_ratio
    # Earning of the last position
    last_position_net_ratio = _calc_position_net_ratio(positions[-1], last_price, price_type)
    position_net_ratios.append(last_position_net_ratio)
    price_change_ratios.append(last_price.actual_price / positions[-1].entry_price.actual_price - 1)
    earning += positions[-1].amount * last_position_net_ratio
    # Price change equals a BUY position, hence `1` instead of `-1`
    price_change = positions[0].amount * 1 * (last_price.actual_price / positions[0].entry_price.actual_price - 1)
    # Win-lose ratio
    wl_ratio = sum([
        # Position net ratios and price change ratios always have the same absolute value but differ in sign,
        # depending on the position type and whether the price is going up or down.
        1 if pnr > pcr else -1 if pnr < pcr else 0
        # Use `pnr if pnr != pcr else 0` to calculate average net ratio
        for pnr, pcr in zip(position_net_ratios, price_change_ratios)
    ]) / len(position_net_ratios)
    return earning, price_change, wl_ratio
