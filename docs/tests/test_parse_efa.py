from datetime import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
from arenkods.common.test_utils import dict_parametrize
from m7_price_predictions.fe.features import parse_efa

london = ZoneInfo("Europe/London")


# For context, parse_efa works fine with any normal EFA block. Where it gets
# tripped up is in EFA 1 which starts at 23:00 and ends at 3:00. This is especially
# so for timezones and daylight savings, where these hours can change.
@dict_parametrize(
    {
        "Non-EFA 1 - returns 2h/4h traded period that isn't EFA 1, same traded period start": {
            "row": pd.Series(data={"date_start": dt(2023, 3, 3, 16, 00, tzinfo=london)}),
            "exp_2h_start": dt(2023, 3, 3, 15, 00, tzinfo=london),
            "exp_4h_start": dt(2023, 3, 3, 15, 00, tzinfo=london),
        },
        "Non-EFA 1 - returns 2h/4h traded period that isn't EFA 1, different traded period start": {
            "row": pd.Series(data={"date_start": dt(2023, 3, 3, 18, 00, tzinfo=london)}),
            "exp_2h_start": dt(2023, 3, 3, 17, 00, tzinfo=london),
            "exp_4h_start": dt(2023, 3, 3, 15, 00, tzinfo=london),
        },
        "23:00 EFA start - Returns 2h/4h traded period that is EFA 1, same traded period start": {
            "row": pd.Series(data={"date_start": dt(2023, 3, 3, 23, 00, tzinfo=london)}),
            "exp_2h_start": dt(2023, 3, 3, 23, 00, tzinfo=london),
            "exp_4h_start": dt(2023, 3, 3, 23, 00, tzinfo=london),
        },
        "00:00 EFA start - Returns 2h/4h traded period that is EFA 1, same traded period start": {
            "row": pd.Series(data={"date_start": dt(2023, 3, 4, 00, 00, tzinfo=london)}),
            "exp_2h_start": dt(2023, 3, 3, 23, 00, tzinfo=london),
            "exp_4h_start": dt(2023, 3, 3, 23, 00, tzinfo=london),
        },
        "01:00 EFA start next day - Returns 2h/4h traded period that is EFA 1, different traded period start": {
            "row": pd.Series(data={"date_start": dt(2023, 3, 4, 1, 00, tzinfo=london)}),
            "exp_2h_start": dt(2023, 3, 4, 1, 00, tzinfo=london),
            "exp_4h_start": dt(2023, 3, 3, 23, 00, tzinfo=london),
        },
        "03:00 EFA start next day - Returns 2h/4h traded period that is EFA 2, same traded period start": {
            "row": pd.Series(data={"date_start": dt(2023, 3, 4, 3, 00, tzinfo=london)}),
            "exp_2h_start": dt(2023, 3, 4, 3, 00, tzinfo=london),
            "exp_4h_start": dt(2023, 3, 4, 3, 00, tzinfo=london),
        },
        "01:00 EFA start next day (SUMMER)- Returns 2h/4h traded period that is EFA 1, different traded period start": {
            "row": pd.Series(data={"date_start": dt(2023, 8, 4, 1, 00, tzinfo=london)}),
            "exp_2h_start": dt(2023, 8, 4, 1, 00, tzinfo=london),
            "exp_4h_start": dt(2023, 8, 3, 23, 00, tzinfo=london),
        },
    }
)
def test_parse_efa(row, exp_2h_start, exp_4h_start):
    result_4h, result_2h = parse_efa(row)
    assert result_2h == exp_2h_start
    assert result_4h == exp_4h_start
