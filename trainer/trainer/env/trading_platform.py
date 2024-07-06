import datetime
import logging
from collections import defaultdict
from enum import IntEnum
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from ..asset.base import DailyAsset, DailyPrice
from .asset_pool import AssetPool, calc_polarity_diff


class PositionType(IntEnum):
    SIDELINE = 0
    BUY = 1
    SELL = 2


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
    metadata = {"render_modes": ["rgb_array"]}

    # Control flags
    is_training: bool

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
    _short_period_penalty: float = 0.0  # Penalty for holding positions for too short a period (UNUSED)

    # Hyperparameters for termination and truncation
    _max_balance_loss: float  # Positive ratio
    _max_balance_gain: float  # Positive ratio
    _max_positions_num: int  # Maximum number of positions (greater than 1) allowed in one episode
    _max_steps_num: int  # Maximum number of steps allowed in one episode

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
        max_balance_loss: float = 0.0,
        max_balance_gain: float = 0.0,
        max_positions_num: int = 0,
        max_steps_num: int = 0,
    ) -> None:
        super().__init__()
        self.is_training = True
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
        self.action_space = gym.spaces.Discrete(len(PositionType))
        # NOTE: Theoretically, we only need the historical price when deciding the position order (buy/sell or hold).
        # However, with DQN (or perhaps many other RL algorithms),
        # to inform the agent about how long the episode has elapsed and how long it takes to finish the episode,
        # knowing "where" the current state is relatively located in an episode is critically required.
        # In my best guess, this can be solved by adding information about current balance or the position we are holding.
        self.observation_space = gym.spaces.Dict({
            # Suppose that delta values (ratios) are greater than -1 and less than 1,
            # meaning prices and other indicators never drop to 0 and never double from previous day.
            "historical_ema_diffs": gym.spaces.Box(-1, 1, shape=(self._historical_days_num,)),
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
            randomizing_start=self.is_training, target_polarity_diff=-self._polarity_diff,
            preferring_secondary=preferring_secondary,
        )
        self._asset.prepare_indicators(close_random_radius=self._CLOSE_RANDOM_RADIUS if self.is_training else None)
        self._date_index = 0
        self._retrieve_prices()
        self._positions = [Position(
            self._prices[-1].date, self.np_random.choice([PositionType.SIDELINE, PositionType.BUY, PositionType.SELL]),
            self._prices[-1].actual_price, self._position_amount,
        )]  # First position
        self._balance = self._initial_balance
        if self._positions[-1].position_type != PositionType.SIDELINE:
            self._balance += self._positions[-1].amount * -self._position_opening_penalty  # Opening penalty
        observation = self._obtain_observation()
        info = {}
        if self.is_training:
            self._date_chosen_counter[self._date_range[self._date_index]][self._asset_symbol] += 1
            self._polarity_diff += calc_polarity_diff(self._prices[-1].price_delta)
        return observation, info

    def step(self, action: np.int64) -> tuple[Dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        reward = 0
        # Recalculate position's net by first reverting net of the previous date.
        # The net of the current date will be calculated later.
        reward -= self._positions[-1].amount * self._last_position_net_ratio
        # Move to a new date (the current date)
        self._date_index += 1
        self._retrieve_prices()
        reward += self._positions[-1].amount * self._last_position_net_ratio  # Position's net of the current date
        if self._positions[-1].position_type != PositionType.SIDELINE:
            reward += self._positions[-1].amount * -self._position_holding_daily_fee  # Holding fee
        # If the position type changes, close the current position and open a new one
        if action != int(self._positions[-1].position_type):
            if self._positions[-1].position_type != PositionType.SIDELINE:
                # Penalize if the previous position is held for too short a period
                reward += self._positions[-1].amount \
                    * -self._short_period_penalty / (self._prices[-1].date - self._positions[-1].date).days
            self._positions.append(Position(
                self._prices[-1].date, PositionType(action),
                self._prices[-1].actual_price, self._position_amount,
            ))
            if action != int(PositionType.SIDELINE):
                reward += self._positions[-1].amount * -self._position_opening_penalty  # Opening penalty
        # Treat the balance as a cumulative reward in each episode
        self._balance += reward
        # Read more about termination and truncation at:
        # - https://gymnasium.farama.org/v0.29.0/tutorials/gymnasium_basics/handling_time_limits/
        # - https://farama.org/Gymnasium-Terminated-Truncated-Step-API
        # Termination conditions
        terminated = (
            # Liquidated
            self._balance < self._initial_balance * (1 - self._max_balance_loss)
        )
        # Truncation conditions
        is_end_of_date = self._date_index >= len(self._date_range) - 1
        truncated = (
            # Reaching the end of training date
            is_end_of_date
            # Realizing profits
            or self._balance >= self._initial_balance * (1 + self._max_balance_gain)
            # Other conditions
            or len(self._positions) >= self._max_positions_num
            or self._date_index >= self._max_steps_num
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
        if self.render_mode != "rgb_array":
            return
        # LINK: `num` and `clear` help prevent memory leak (See: https://stackoverflow.com/a/65910539)
        # `num=1` is reserved for trading chart
        plt.rcParams.update({"font.size": 8})
        figure = plt.figure(figsize=(10, 6 if self.is_training else 3), dpi=800, num=1, clear=True)
        figure.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9)
        # Plot prices and positions
        axes = figure.add_subplot(211 if self.is_training else 111)
        prices = self._asset.retrieve_historical_prices(
            self._date_range[self._date_index],
            # Retrieve all the days before the current date in the date range
            self._date_index + self._historical_days_num,
            randomizing_end=self.is_training,
        )
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
                    axes.plot([d for d, _ in date_prices], [p for _, p in date_prices], color=color, linewidth=0.5)
                    axes.plot(*date_prices[0], color=color, marker="o", markersize=0.5)
                # Move to next position
                position_index = next_position_index
                date_prices = [date_price]
        # Plot date counter
        dates = [p.date for p in prices]
        if self.is_training:
            axes = figure.add_subplot(212)
            axes.bar(dates, [sum(self._date_chosen_counter[d].values()) for d in dates])
        # Draw figure
        figure.canvas.draw()
        image = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8) \
            .reshape(figure.canvas.get_width_height()[::-1] + (3,))
        return image

    def refresh(self):
        self._polarity_diff = 0
        self._date_chosen_counter = defaultdict(lambda: defaultdict(int))

    @property
    def _asset(self) -> DailyAsset:
        return self._asset_pool.get_asset(self._asset_symbol)

    @property
    def _position_amount(self) -> float:
        return self._POSITION_AMOUNT_UNIT if self.is_training else self._prices[-1].actual_price

    @property
    def _initial_balance(self) -> float:
        # NOTE: A zero initial balance may seem illogical, but in evaluation, only earnings matter, not the initial balance.
        # The balance is mainly used for training, while using price as a position amount often implies evaluation.
        return self._INITIAL_BALANCE_UNIT if self.is_training else 0

    @property
    def _last_position_net_ratio(self) -> float:
        return calc_position_net_ratio(self._positions[-1], self._prices[-1].actual_price)

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
            "historical_ema_diffs": np.array(self._minmax_scale([p.ema_diff for p in self._prices])),
        }

    @staticmethod
    def _minmax_scale(deltas: List[float]) -> List[float]:
        # TODO: Is it ok to use instance normalization?
        max_delta = max([abs(d) for d in deltas])
        deltas = [d / max_delta for d in deltas]
        return deltas


def calc_position_net_ratio(position: Position, actual_price: float) -> float:
    position_type = position.position_type
    if position_type == PositionType.SIDELINE:
        return 0
    else:
        return (actual_price / position.entry_price - 1) * (1 if position_type == PositionType.BUY else -1)


def calc_earning(
    positions: List[Position], final_price: DailyPrice,
    position_holding_daily_fee: float = 0.0, position_opening_penalty: float = 0.0, short_period_penalty: float = 0.0,
) -> Tuple[float, float]:
    if len(positions) < 1 or positions[-1].date > final_price.date:
        return (0, 0)
    earning = 0
    # Rewards (net, fee, penalty) of closed positions (excluding the last position, as it is probably not yet closed)
    for prev_position, cur_position in zip(positions[:-1], positions[1:]):
        # Position net
        earning += prev_position.amount * calc_position_net_ratio(prev_position, cur_position.entry_price)
        if prev_position.position_type != PositionType.SIDELINE:
            # Holding fee
            earning += (cur_position.date - prev_position.date).days \
                * prev_position.amount * -position_holding_daily_fee
            # TODO: Skip calculating penalties when evaluating
            # Opening penalty
            earning += prev_position.amount * -position_opening_penalty
            # Short-period penalty
            earning += prev_position.amount * -short_period_penalty / (cur_position.date - prev_position.date).days
        logging.debug("%s %f %s", prev_position.date, prev_position.entry_price, prev_position.position_type)
    # Reward of the last position
    earning += positions[-1].amount * calc_position_net_ratio(positions[-1], final_price.actual_price)
    if positions[-1].position_type != PositionType.SIDELINE:
        earning += (final_price.date - positions[-1].date).days * positions[-1].amount * -position_holding_daily_fee
        earning += positions[-1].amount * -position_opening_penalty
    logging.debug("%s %f %s", positions[-1].date, positions[-1].entry_price, positions[-1].position_type)
    # Actual price change equals a BUY position, hence `1` instead of `-1`
    actual_price_change = (final_price.actual_price / positions[0].entry_price - 1) * 1 * positions[0].amount
    return earning, actual_price_change
