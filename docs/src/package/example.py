import os
import sys
from datetime import datetime as dt
from datetime import timezone as tz


def add(a: int, b: int) -> int:
    return a + b


class GoodName:
    def __init__(self):
        self.something = os.environ.get("SOMETHING", "default")
        self.something2 = sys.argv
        self.now = dt.now(tz=tz.utc)
