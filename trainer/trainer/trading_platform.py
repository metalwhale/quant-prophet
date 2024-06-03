import datetime
import logging
from enum import IntEnum
from typing import Any, Dict, List, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import markers

from .asset import DailyAsset, Price


class PositionType(IntEnum):
    LONG = 0
    SHORT = 1


class Position:
    _time: datetime.datetime
    _amount: float
    _position_type: PositionType

    def __init__(self, time: datetime.datetime, amount: float, position_type: PositionType) -> "Position":
        self._time = time
        self._amount = amount
        self._position_type = position_type

    @property
    def time(self) -> datetime.datetime:
        return self._time

    @property
    def amount(self) -> float:
        return self._amount

    @property
    def position_type(self) -> PositionType:
        return self._position_type


# Doc:
# - https://gymnasium.farama.org/v0.29.0/tutorials/gymnasium_basics/environment_creation/
# - https://stable-baselines3.readthedocs.io/en/v2.3.2/guide/custom_env.html
class TradingPlatform(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Terminology
    # "Episode" and "step":
    # - "Episode" and "step" are terms usually used in reinforcement learning.
    #   In our case, an "episode" consists of several consecutive orders of positions,
    #   and a "step" simply refers to a tradable day.
    # "Date":
    # - A "date" refers to a "tradable day of the date type", i.e., a day on which we can open an position.
    #   We use "date" as the end date of a date range when retrieving historical data.
    #   Note that, unless otherwise explained, we consider a "date" to have this meaning.
    # - An example of a date that is not a "date" according to our definition is the "first date" of an asset,
    #   and all dates before the "initial date" of an episode (see the `reset` method for details),
    #   because they are not used as the end date when retrieving historical data or for opening positions,
    #   but rather as serial data used for training the model.
    #
    # Example of an episode for a specific asset:
    #   (Note: `~~~~~~~~*` depicts length of `_historical_days_num`,
    #    with `*` being the date used as the end date when retrieving the historical data)
    #
    #   |<---Unused--->|<--------------------Training data---------------------->|<-------------Validation data--------------->|
    #   |              |                                                         |                                             |
    #   o======================o=================================================o=============================================o
    #   |              |       |                                  |              |
    #   |              |       |                                  |              Last training date (`_last_training_date`)
    #   |              |       |                          ~~~~~~~~*
    #   |              |       |                                  The "date" will continuously grow until the last training date
    #   |              ~~~~~~~~*
    #   |                      Episode's "initial date" (`_initial_date`), randomly chosen within range
    #   |                      from the asset's "first date" plus number of historical data, to the day before last training date
    #   |
    #   |<-------------------->| All dates before "initial date" are non-tradable,
    #   |                      | and have at least the length of `_historical_days_num`
    #   |
    #   Asset's "first date" (`asset_first_date`)

    # Hyperparameters
    _asset_pool: List[DailyAsset]
    _historical_days_num: int  # Number of days used for retrieving historical data
    _last_training_date: datetime.date
    # TODO: Reconsider the meaning of the opening fee.
    # I believe that changing the opening fee affects how often new positions are opened,
    # i.e., increasing the opening fee means the model may learn to open fewer positions.
    _position_opening_fee: float  # Positive ratio
    # TODO: Reconsider the meaning of the daily fee.
    # Its sole purpose currently seems to be only preventing holding a position too long, causing a loss before earning.
    # Does it still make sense since we are always holding every day?
    _position_holding_daily_fee: float  # Positive ratio
    _max_balance_loss: float  # Positive ratio
    _min_positions_num: int  # Minimum number of positions (greater than 1) allowed in one episode
    _min_steps_num: int  # Minimum number of steps allowed in one episode

    # State components
    _asset_index: int  # Changed only if we reset
    _initial_date: datetime.date  # The random initial date of each episode
    _date: datetime.date  # Grows in the same episode, resets to initial date for a new episode
    _prices: List[Price]  # Updated whenever the date changes
    _positions: List[Position]  # Keeps adding positions in the same episode, clears them all for a new episode
    _balance: float  # Resets to initial balance after each episode

    # Constants
    _POSITION_AMOUNT: float = 1  # Equal to or less than the initial balance
    _INITIAL_BALANCE: float = 1

    def __init__(
        self,
        asset_pool: List[DailyAsset],
        historical_days_num: int,
        last_training_date: datetime.date,
        position_opening_fee: float,
        position_holding_daily_fee: float,
        max_balance_loss: float,
        min_positions_num: int = 1,
        min_steps_num: int = 1,
    ) -> "TradingPlatform":
        super().__init__()
        # Hyperparameters
        self._asset_pool = asset_pool
        self._historical_days_num = historical_days_num
        self._last_training_date = last_training_date  # TODO: Check if there are enough days to retrieve historical data
        self._position_opening_fee = position_opening_fee
        self._position_holding_daily_fee = position_holding_daily_fee
        self._max_balance_loss = max_balance_loss
        self._min_positions_num = min_positions_num
        self._min_steps_num = min_steps_num
        # State components
        self._prices = []
        # Environment
        self.action_space = gym.spaces.Discrete(len(PositionType))
        self.observation_space = gym.spaces.Dict({
            # Suppose that price deltas (ratio) are greater than -1 and less than 1,
            # meaning price never drops to 0 and never doubles from previous day.
            "historical_price_deltas": gym.spaces.Box(-1, 1, shape=(self._historical_days_num,)),
            # Position types have the same values as action space.
            "position_type": gym.spaces.Discrete(len(PositionType)),
            # Similar to price deltas, suppose that position net ratio is in range (-1, 1) compared to entry price.
            # TODO: Consider cases when position net ratio can be greater than 1.
            "position_net_ratio": gym.spaces.Box(-1, 1, shape=(1,)),
        })

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._asset_index = self.np_random.integers(0, high=len(self._asset_pool))
        asset_first_date = self._asset.get_first_date()
        # Randomly choose a date within range from the asset's first date plus number of historical data
        # to the day before last training date, and consider it as end date for retrieving historical prices.
        self._initial_date = asset_first_date \
            + datetime.timedelta(days=float(self.np_random.integers(
                self._historical_days_num - 1,
                high=(self._last_training_date - asset_first_date).days,  # Exclusive
            )))
        self._date = self._initial_date
        self._retrieve_prices()
        self._positions = [Position(
            self._prices[-1].time, self._POSITION_AMOUNT,
            self.np_random.choice([PositionType.LONG, PositionType.SHORT]),
        )]  # First position
        self._balance = self._INITIAL_BALANCE \
            + self._positions[-1].amount * -self._position_opening_fee  # Opening fee
        observation = self._obtain_observation()
        info = {}
        return observation, info

    def step(self, action: np.int64) -> tuple[Dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        self._date += datetime.timedelta(days=1)
        self._retrieve_prices()
        reward = self._positions[-1].amount * -self._position_holding_daily_fee  # Holding fee
        # If the position type changes, close the current position and open a new one
        if action != int(self._positions[-1].position_type):
            # Previous position's net
            reward += self._positions[-1].amount * self._last_position_net_ratio
            self._positions.append(Position(self._prices[-1].time, self._POSITION_AMOUNT, PositionType(action)))
            reward += self._positions[-1].amount * -self._position_opening_fee  # Opening fee
        # Last position's net
        last_postion_net = self._positions[-1].amount * self._last_position_net_ratio
        # Termination condition
        terminated = (
            # Liquidated
            self._balance + reward + last_postion_net < self._INITIAL_BALANCE * (1 - self._max_balance_loss)
            # Normally finished the episode without being forced to quit
            or (
                len(self._positions) >= self._min_positions_num
                # Number of steps equals the number of dates between the date of first position and the current date.
                and (self._date - self._positions[0].time.date()).days >= self._min_steps_num
            )
        )
        # Truncation condition
        truncated = self._date >= self._last_training_date
        if terminated or truncated:
            reward += last_postion_net
        # Treat the balance as a cumulative reward in each episode
        self._balance += reward
        observation = self._obtain_observation()
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self) -> Any | List[Any] | None:
        if self.render_mode != "rgb_array":
            return
        figure = plt.figure()
        axes = figure.add_subplot(111)
        prices = self._asset.retrieve_historical_prices(
            self._prices[-1].time,  # Use the exact datetime for the last price
            (self._date - self._initial_date).days + self._historical_days_num,
        )
        axes.plot(
            [p.time for p in prices],
            [p.actual_price for p in prices],
        )
        for position in self._positions:
            is_long = position.position_type == PositionType.LONG
            axes.plot(
                position.time, self._retrieve_price(position.time),
                marker=markers.CARETUP if is_long else markers.CARETDOWN,
                color="green" if is_long else "red",
            )
        figure.canvas.draw()
        image = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8) \
            .reshape(figure.canvas.get_width_height()[::-1] + (3,))
        plt.close(figure)
        return image

    def calc_earning(self, positions: List[Position]) -> Tuple[float, float]:
        if len(positions) < 2:
            return (0, 0)
        earning = 0
        # Nets and fees of closed positions (excluding the last position, as it is probably not yet closed)
        for prev_position, cur_position in zip(positions[:-1], positions[1:]):
            prev_price = self._retrieve_price(prev_position.time)
            cur_price = self._retrieve_price(cur_position.time)
            # Opening fee
            earning += prev_position.amount * -self._position_opening_fee
            # Holding fee
            earning += (cur_position.time.date() - prev_position.time.date()).days \
                * prev_position.amount * -self._position_holding_daily_fee
            # Position Earning
            earning += (cur_price / prev_price - 1) * (1 if prev_position.position_type == PositionType.LONG else -1) \
                * prev_position.amount
            logging.debug("%s %f %s", prev_position.time, prev_price, prev_position.position_type)
        # Last position plays a closing role and doesn't contribute to the earning
        logging.debug(
            "%s %f %s",
            positions[-1].time, self._retrieve_price(positions[-1].time), positions[-1].position_type,
        )
        # Actual price change
        price_change = (self._retrieve_price(positions[-1].time) - self._retrieve_price(positions[0].time)) \
            * 1 * positions[0].amount  # Pure price change equals a LONG position, hence `1` instead of `-1`
        return earning, price_change

    @property
    def _last_position_net_ratio(self) -> float:
        return (self._prices[-1].actual_price / self._retrieve_price(self._positions[-1].time) - 1) \
            * (1 if self._positions[-1].position_type == PositionType.LONG else -1)

    @property
    def _asset(self) -> DailyAsset:
        return self._asset_pool[self._asset_index]

    def _retrieve_price(self, time: datetime.datetime) -> float:
        return self._asset.retrieve_historical_prices(time, 1)[-1].actual_price

    def _retrieve_prices(self):
        # Since `retrieve_historical_prices` chooses the end price from a random time on the same end date,
        # multiple calls produce different results for the end price.
        # We need to avoid calling it again if the current date matches when we previously retrieved prices.
        if len(self._prices) > 0 and self._prices[-1].time.date() == self._date:
            return
        self._prices = self._asset.retrieve_historical_prices(self._date, self._historical_days_num)

    def _obtain_observation(self) -> Dict[str, Any]:
        return {
            "historical_price_deltas": [p.price_delta for p in self._prices],
            "position_type": self._positions[-1].position_type,
            "position_net_ratio": self._last_position_net_ratio,
        }
