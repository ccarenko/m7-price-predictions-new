from datetime import datetime as dt
from datetime import timezone as tz
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from arenkods.common.test_utils import dict_parametrize
from m7_price_predictions.fe.features import engineer_auction_shaping, format_trades
from m7_price_predictions.schemas import schemas

london_tz = ZoneInfo("Europe/London")

test_data = {
    "All m7 products available": {
        "trades_data": pd.DataFrame(
            {
                "date_start": 3 * [dt(2023, 8, 7, 1, 30, tzinfo=tz.utc), dt(2023, 8, 7, 2, 0, tzinfo=tz.utc)],
                "label": [
                    "m7_public_trades_2h",
                    "m7_public_trades_2h",
                    "m7_public_trades_4h",
                    "m7_public_trades_4h",
                    "m7_public_trades_hh",
                    "m7_public_trades_hh",
                ],
                "traded_price": [40.0, 40.0, 60.0, 60.0, np.nan, 30.0],
                "date_on": 3 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc), dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)],
            }
        ),
        "hourly_trades": pd.DataFrame(
            {
                "date_on": sorted(2 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc), dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)]),
                "label": 4 * ["ets_hour_price"],
                "date_start": 2 * [dt(2023, 8, 7, 1, 0, tzinfo=tz.utc), dt(2023, 8, 7, 2, 0, tzinfo=tz.utc)],
                "date_end": 2 * [dt(2023, 8, 7, 2, 0, tzinfo=tz.utc), dt(2023, 8, 7, 3, 0, tzinfo=tz.utc)],
                "value": [60.0, 70.0, 60.0, 70.0],
            }
        ),
        "format_trades_df": pd.DataFrame(
            {
                "date_on": 3 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc)] + 2 * [dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)],
                "date_start": [
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, 30, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                ],
                "ets_hour_price": [60.0, np.nan, 70.0, 60.0, 70.0],
                "m7_public_trades_2h": [np.nan, 40.0, np.nan, np.nan, 40.0],
                "m7_public_trades_4h": [np.nan, 60.0, np.nan, np.nan, 60.0],
                "m7_public_trades_hh": [np.nan, np.nan, np.nan, np.nan, 30.0],
                "4h_start": 5 * [dt(2023, 8, 6, 23, tzinfo=london_tz)],
                "2h_start": 5 * [dt(2023, 8, 7, 1, tzinfo=london_tz)],
            }
        ),
        "shaping_df": pd.DataFrame(
            {
                "date_on": 3 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc)] + 2 * [dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)],
                "date_start": [
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, 30, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                ],
                "ets_hour_price": [60.0, 60.0, 70.0, 60.0, 70.0],
                "traded_price": [np.nan, 46.67, np.nan, np.nan, 30.00],
            }
        ),
    },
    "m7_public_trades_2h missing": {
        "trades_data": pd.DataFrame(
            {
                "date_start": 2 * [dt(2023, 8, 7, 1, 30, tzinfo=tz.utc), dt(2023, 8, 7, 2, 0, tzinfo=tz.utc)],
                "label": ["m7_public_trades_4h", "m7_public_trades_4h", "m7_public_trades_hh", "m7_public_trades_hh"],
                "traded_price": [40.0, 40.0, 30.0, 30.0],
                "date_on": 2 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc), dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)],
            }
        ),
        "hourly_trades": pd.DataFrame(
            {
                "date_on": sorted(2 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc), dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)]),
                "label": 4 * ["ets_hour_price"],
                "date_start": 2 * [dt(2023, 8, 7, 1, 0, tzinfo=tz.utc), dt(2023, 8, 7, 2, 0, tzinfo=tz.utc)],
                "date_end": 2 * [dt(2023, 8, 7, 2, 0, tzinfo=tz.utc), dt(2023, 8, 7, 3, 0, tzinfo=tz.utc)],
                "value": [60.0, 70.0, 60.0, 70.0],
            }
        ),
        "format_trades_df": pd.DataFrame(
            {
                "date_on": 3 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc)] + 2 * [dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)],
                "date_start": [
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, 30, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                ],
                "ets_hour_price": [60.0, np.nan, 70.0, 60.0, 70.0],
                "m7_public_trades_2h": 5 * np.nan,
                "m7_public_trades_4h": [np.nan, 40.0, np.nan, np.nan, 40.0],
                "m7_public_trades_hh": [np.nan, 30.0, np.nan, np.nan, 30.0],
                "4h_start": 5 * [dt(2023, 8, 6, 23, tzinfo=london_tz)],
                "2h_start": 5 * [dt(2023, 8, 7, 1, tzinfo=london_tz)],
            }
        ),
        "shaping_df": pd.DataFrame(
            {
                "date_on": 3 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc)] + 2 * [dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)],
                "date_start": [
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, 30, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                ],
                "ets_hour_price": [60.0, 60.0, 70.0, 60.0, 70.0],
                "traded_price": [np.nan, 30.0, np.nan, np.nan, 30.0],
            }
        ),
    },
    "m7_public_trades_4h missing": {
        "trades_data": pd.DataFrame(
            {
                "date_start": 2 * [dt(2023, 8, 7, 1, 30, tzinfo=tz.utc), dt(2023, 8, 7, 2, 0, tzinfo=tz.utc)],
                "label": ["m7_public_trades_2h", "m7_public_trades_2h", "m7_public_trades_hh", "m7_public_trades_hh"],
                "traded_price": [50.0, 50.0, 35.0, 35.0],
                "date_on": 2 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc), dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)],
            }
        ),
        "hourly_trades": pd.DataFrame(
            {
                "date_on": sorted(2 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc), dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)]),
                "label": 4 * ["ets_hour_price"],
                "date_start": 2 * [dt(2023, 8, 7, 1, 0, tzinfo=tz.utc), dt(2023, 8, 7, 2, 0, tzinfo=tz.utc)],
                "date_end": 2 * [dt(2023, 8, 7, 2, 0, tzinfo=tz.utc), dt(2023, 8, 7, 3, 0, tzinfo=tz.utc)],
                "value": [60.0, 70.0, 60.0, 70.0],
            }
        ),
        "format_trades_df": pd.DataFrame(
            {
                "date_on": 3 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc)] + 2 * [dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)],
                "date_start": [
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, 30, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                ],
                "ets_hour_price": [60.0, np.nan, 70.0, 60.0, 70.0],
                "m7_public_trades_2h": [np.nan, 50.0, np.nan, np.nan, 50.0],
                "m7_public_trades_4h": 5 * [np.nan],
                "m7_public_trades_hh": [np.nan, 35.0, np.nan, np.nan, 35.0],
                "4h_start": 5 * [dt(2023, 8, 6, 23, tzinfo=london_tz)],
                "2h_start": 5 * [dt(2023, 8, 7, 1, tzinfo=london_tz)],
            }
        ),
        "shaping_df": pd.DataFrame(
            {
                "date_on": 3 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc)] + 2 * [dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)],
                "date_start": [
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, 30, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                ],
                "ets_hour_price": [60.0, 60.0, 70.0, 60.0, 70.0],
                "traded_price": [np.nan, 35.0, np.nan, np.nan, 35.0],
            }
        ),
    },
    "m7_public_trades_4h and 2h missing": {
        "trades_data": pd.DataFrame(
            {
                "date_start": [dt(2023, 8, 7, 1, 30, tzinfo=tz.utc), dt(2023, 8, 7, 2, 0, tzinfo=tz.utc)],
                "label": ["m7_public_trades_hh", "m7_public_trades_hh"],
                "traded_price": [35.0, 35.0],
                "date_on": [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc), dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)],
            }
        ),
        "hourly_trades": pd.DataFrame(
            {
                "date_on": sorted(2 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc), dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)]),
                "label": 4 * ["ets_hour_price"],
                "date_start": 2 * [dt(2023, 8, 7, 1, 0, tzinfo=tz.utc), dt(2023, 8, 7, 2, 0, tzinfo=tz.utc)],
                "date_end": 2 * [dt(2023, 8, 7, 2, 0, tzinfo=tz.utc), dt(2023, 8, 7, 3, 0, tzinfo=tz.utc)],
                "value": [60.0, 70.0, 60.0, 70.0],
            }
        ),
        "format_trades_df": pd.DataFrame(
            {
                "date_on": 3 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc)] + 2 * [dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)],
                "date_start": [
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, 30, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                ],
                "ets_hour_price": [60.0, np.nan, 70.0, 60.0, 70.0],
                "m7_public_trades_2h": 5 * [np.nan],
                "m7_public_trades_4h": 5 * [np.nan],
                "m7_public_trades_hh": [np.nan, 35.0, np.nan, np.nan, 35.0],
                "4h_start": 5 * [dt(2023, 8, 6, 23, tzinfo=london_tz)],
                "2h_start": 5 * [dt(2023, 8, 7, 1, tzinfo=london_tz)],
            }
        ),
        "shaping_df": pd.DataFrame(
            {
                "date_on": 3 * [dt(2023, 8, 7, 0, 21, tzinfo=tz.utc)] + 2 * [dt(2023, 8, 7, 0, 51, tzinfo=tz.utc)],
                "date_start": [
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, 30, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                    dt(2023, 8, 7, 1, tzinfo=tz.utc),
                    dt(2023, 8, 7, 2, tzinfo=tz.utc),
                ],
                "ets_hour_price": [60.0, 60.0, 70.0, 60.0, 70.0],
                "traded_price": [np.nan, 35.0, np.nan, np.nan, 35.0],
            }
        ),
    },
}


@dict_parametrize(test_data)
def test_trades_transforms(trades_data, hourly_trades, format_trades_df, shaping_df):
    got1 = format_trades(trades_data, hourly_trades)
    pd.testing.assert_frame_equal(schemas.FormattedTradesDF.validate(format_trades_df), got1)
    got2 = engineer_auction_shaping(format_trades_df)
    pd.testing.assert_frame_equal(schemas.ShapedTradesDF.validate(shaping_df), got2)
