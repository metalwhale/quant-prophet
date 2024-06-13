import datetime
import logging
from enum import IntEnum
from typing import Any, Dict, List, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import markers

from ..asset.base import DailyAsset, DailyPrice
from .asset_pool import AssetPool, calc_polarity_diff


class PositionType(IntEnum):
    LONG = 0
    SHORT = 1


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

    _is_training_mode: bool

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
    #   |                                                                         Last training date (`_last_training_date`)
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

    # Hyperparameters
    _asset_pool: AssetPool
    _historical_days_num: int  # Number of days used for retrieving historical data
    _last_training_date: datetime.date
    # TODO: Reconsider the meaning of the opening fee.
    # I believe that changing the opening fee affects how often new positions are opened,
    # i.e., increasing the opening fee means the model may learn to open fewer positions.
    _position_opening_fee: float  # Positive ratio
    # TODO: Reconsider the meaning of the daily fee.
    # Its sole purpose currently seems to be only preventing holding a position too long, causing a loss before earning.
    # Does it still make sense since we are always holding every day?
    _position_holding_daily_fee: float = 0.0  # Positive ratio (UNUSED)
    _short_period_penalty: float = 0.0  # Penalty for holding positions for too short a period (UNUSED)
    _max_balance_loss: float  # Positive ratio
    _min_positions_num: int  # Minimum number of positions (greater than 1) allowed in one episode
    _min_steps_num: int  # Minimum number of steps allowed in one episode

    # State components
    _asset_symbol: str  # Changed only if we reset
    _date_range: List[datetime.date]  # The random date range of each episode
    _date_index: int  # Grows in the same episode, resets to 0 for a new episode
    _prices: List[DailyPrice]  # Updated whenever the date changes
    _polarity_diff: int
    _positions: List[Position]  # Keeps adding positions in the same episode, clears them all for a new episode
    _balance: float  # Resets to initial balance after each episode

    # Constants, mainly used only for training
    # NOTE: In reality, initial balance should be higher than position amount to cover opening fees.
    # Here, all set to 1 for simplicity.
    _POSITION_AMOUNT_UNIT: float = 1  # Equal to or less than the initial balance
    _INITIAL_BALANCE_UNIT: float = 1

    def __init__(
        self,
        asset_pool: AssetPool,
        historical_days_num: int,
        last_training_date: datetime.date,
        position_opening_fee: float = 0.0,
        max_balance_loss: float = 0.0,
        min_positions_num: int = 0,
        min_steps_num: int = 0,
    ) -> None:
        super().__init__()
        # Hyperparameters
        self._asset_pool = asset_pool
        self._historical_days_num = historical_days_num
        self._last_training_date = last_training_date  # TODO: Check if there are enough days to retrieve historical data
        # These following hyperparameters are mainly used only for training,
        # by calculating reward and determining whether to terminate an episode.
        self._position_opening_fee = position_opening_fee
        self._max_balance_loss = max_balance_loss
        self._min_positions_num = min_positions_num
        self._min_steps_num = min_steps_num
        # State components
        self._prices = []
        self._polarity_diff = 0
        # Environment
        self.action_space = gym.spaces.Discrete(len(PositionType))
        self.observation_space = gym.spaces.Dict({
            # Suppose that price deltas (ratio) are greater than -1 and less than 1,
            # meaning price never drops to 0 and never doubles from previous day.
            "historical_price_deltas": gym.spaces.Box(-1, 1, shape=(self._historical_days_num,)),
        })
        # Mode
        self.is_training_mode = True

    def reset(
        self, *,
        seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._asset_symbol, self._date_range = self._asset_pool.choose_asset_date(
            random_start_day=self.is_training_mode, target_polarity_diff=-self._polarity_diff,
        )
        self._date_index = 0
        self._retrieve_prices()
        self._positions = [Position(
            self._prices[-1].date, self.np_random.choice([PositionType.LONG, PositionType.SHORT]),
            self._prices[-1].actual_price, self._position_amount,
        )]  # First position
        self._balance = self._initial_balance + self._positions[-1].amount * -self._position_opening_fee  # Opening fee
        observation = self._obtain_observation()
        info = {}
        return observation, info

    def step(self, action: np.int64) -> tuple[Dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        reward = 0
        # Recalculate position's net by first reverting net of the previous date.
        # The net of the current date will be calculated later.
        reward -= self._positions[-1].amount * self._last_position_net_ratio
        # Move to a new date (the current date)
        self._date_index += 1
        self._retrieve_prices()
        reward += self._positions[-1].amount * -self._position_holding_daily_fee  # Holding fee
        reward += self._positions[-1].amount * self._last_position_net_ratio  # Position's net of the current date
        # If the position type changes, close the current position and open a new one
        if action != int(self._positions[-1].position_type):
            # Penalize if the previous position is held for too short a period
            reward += self._positions[-1].amount \
                * -self._short_period_penalty / (self._prices[-1].date - self._positions[-1].date).days
            self._positions.append(Position(
                self._prices[-1].date, PositionType(action),
                self._prices[-1].actual_price, self._position_amount,
            ))
            reward += self._positions[-1].amount * -self._position_opening_fee  # Opening fee
        # Termination condition
        terminated = (
            # Liquidated
            self._balance + reward < self._initial_balance * (1 - self._max_balance_loss)
            # Normally finished the episode without being forced to quit
            or (len(self._positions) >= self._min_positions_num and self._date_index >= self._min_steps_num)
        )
        # Truncation condition
        truncated = self._date_index >= len(self._date_range) - 1
        # Treat the balance as a cumulative reward in each episode
        self._balance += reward
        observation = self._obtain_observation()
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self) -> Any | List[Any] | None:
        if self.render_mode != "rgb_array":
            return
        figure = plt.figure(figsize=(10, 6), dpi=200)
        figure.subplots_adjust(left=0.05, bottom=0.05, right=1, top=1)
        axes = figure.add_subplot(111)
        prices = self._asset.retrieve_historical_prices(
            self._date_range[self._date_index],
            # Retrieve all the days before the current date in the date range
            self._date_index + self._historical_days_num,
            indeterministic=self.is_training_mode,
        )
        axes.plot(
            [p.date for p in prices],
            [p.actual_price for p in prices],
        )
        for position in self._positions:
            is_long = position.position_type == PositionType.LONG
            axes.plot(
                position.date, position.entry_price,
                marker=markers.CARETUP if is_long else markers.CARETDOWN,
                color="green" if is_long else "red",
            )
        figure.canvas.draw()
        image = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8) \
            .reshape(figure.canvas.get_width_height()[::-1] + (3,))
        plt.close(figure)
        return image

    @property
    def is_training_mode(self) -> bool:
        return self._is_training_mode

    @is_training_mode.setter
    def is_training_mode(self, is_training_mode: bool):
        self._is_training_mode = is_training_mode
        self._asset_pool.apply_date_range_matcher(
            self._min_steps_num,
            self._historical_days_num,
            min_date=self._last_training_date if not self.is_training_mode else None,
            max_date=self._last_training_date if self.is_training_mode else None,
        )

    @property
    def _asset(self) -> DailyAsset:
        return self._asset_pool.get_asset(self._asset_symbol)

    @property
    def _position_amount(self) -> float:
        return self._POSITION_AMOUNT_UNIT if self.is_training_mode else self._prices[-1].actual_price

    @property
    def _initial_balance(self) -> float:
        # NOTE: A zero initial balance may seem illogical, but in evaluation, only earnings matter, not the initial balance.
        # The balance is mainly used for training, while using price as a position amount often implies evaluation.
        return self._INITIAL_BALANCE_UNIT if self.is_training_mode else 0

    @property
    def _last_position_net_ratio(self):
        return calc_position_net_ratio(self._positions[-1], self._prices[-1].actual_price)

    # Should be called right after updating `self._date_index` to the newest date
    def _retrieve_prices(self):
        # Since `retrieve_historical_prices` chooses the end price from a random time on the same end date,
        # multiple calls produce different results for the end price.
        # We need to avoid calling it again if the current date matches when we previously retrieved prices.
        date = self._date_range[self._date_index]
        if len(self._prices) > 0 and self._prices[-1].date == date:
            return
        self._prices = self._asset.retrieve_historical_prices(
            date, self._historical_days_num,
            indeterministic=self.is_training_mode,
        )
        self._polarity_diff += calc_polarity_diff(self._prices[-1].price_delta)

    def _obtain_observation(self) -> Dict[str, Any]:
        # See: https://stackoverflow.com/questions/73922332/dict-observation-space-for-stable-baselines3-not-working
        return {
            "historical_price_deltas": np.array([p.price_delta for p in self._prices]),
        }


def calc_position_net_ratio(position: Position, actual_price: float) -> float:
    return (actual_price / position.entry_price - 1) * (1 if position.position_type == PositionType.LONG else -1)


def calc_earning(
    positions: List[Position], final_price: DailyPrice,
    position_opening_fee: float = 0.0, position_holding_daily_fee: float = 0.0, short_period_penalty: float = 0.0,
) -> Tuple[float, float]:
    if len(positions) < 1 or positions[-1].date > final_price.date:
        return (0, 0)
    earning = 0
    # Nets and fees of closed positions (excluding the last position, as it is probably not yet closed)
    for prev_position, cur_position in zip(positions[:-1], positions[1:]):
        # TODO: Skip calculating penalties when evaluating
        # Short-period penalty
        earning += prev_position.amount * -short_period_penalty / (cur_position.date - prev_position.date).days
        # Opening fee
        earning += prev_position.amount * -position_opening_fee
        # Holding fee
        earning += (cur_position.date - prev_position.date).days * prev_position.amount * -position_holding_daily_fee
        # Position net
        earning += prev_position.amount * calc_position_net_ratio(prev_position, cur_position.entry_price)
        logging.debug("%s %f %s", prev_position.date, prev_position.entry_price, prev_position.position_type)
    # Net and fee of the last position
    earning += positions[-1].amount * -position_opening_fee
    earning += (final_price.date - positions[-1].date).days * positions[-1].amount * -position_holding_daily_fee
    earning += positions[-1].amount * calc_position_net_ratio(positions[-1], final_price.actual_price)
    logging.debug("%s %f %s", positions[-1].date, positions[-1].entry_price, positions[-1].position_type)
    # Actual price change equals a LONG position, hence `1` instead of `-1`
    actual_price_change = (final_price.actual_price / positions[0].entry_price - 1) * 1 * positions[0].amount
    return earning, actual_price_change
