import datetime


OPENING_HOUR: int = 8
CLOSING_HOUR: int = 16


def find_min_tradable_start_date(time: datetime.datetime) -> datetime.date:
    # Add 1 day if initial time is after closing hour
    # And add 1 more day for price deltas as they require a previous day for calculation
    return (time + datetime.timedelta(days=1 if time.hour < CLOSING_HOUR else 2)).date()
