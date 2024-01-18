import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime as dt
from datetime import timedelta as td
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import orjson
import pandas as pd
from arenkods.data.fs import FeatureStore
from arenkods.interval import period
from arenkods.interval.constant import SETTLEMENT_PERIOD_LENGTH as SPL
from m7_price_predictions.m7_settings import settings
from m7_price_predictions.schemas import schemas


def add_engineered_features(raw_data: pd.DataFrame, trades_data: pd.DataFrame, date_on: Iterable) -> pd.DataFrame:
    dfs = [raw_data]
    df_po = get_order_stacks(raw_data)
    if df_po is not None and not df_po.empty:
        dfs.append(df_po)
    df_m7 = get_m7_products(
        df=trades_data,
        hourly_trades=raw_data[raw_data["label"] == "ets_hour_price"],
        date_on=date_on,
    )
    if df_m7 is not None and not df_m7.empty:
        dfs.append(df_m7)

    df_out = pd.concat(dfs, axis=0)
    df_out = df_out[df_out["label"] != "m7_orderbook_hh"].drop_duplicates(
        subset=["date_on", "date_start", "date_end", "label"]
    )

    return df_out


def get_order_stacks(df: pd.DataFrame) -> pd.DataFrame | None:
    if df["label"].eq("m7_orderbook_hh").any():
        df_po = df[df["label"] == "m7_orderbook_hh"]
        # the "value" column is a nested dictionary, so we need to explode it
        # and then rename the columns to match the names in the list
        # i.e., exploding the column would return "pressure.-10", we want "-10 pressure".
        df_values = pd.json_normalize(df_po["value"]).rename(
            columns=lambda x: x.split(".")[1] + " pressure" if "pressure" in x else x
        )[settings.eng_features]

        df_po = pd.concat((df_po.reset_index(drop=True), df_values), axis=1).drop(columns=["label", "value"])
        df_po = pd.melt(
            df_po,
            id_vars=["date_on", "date_start", "date_end"],
            value_vars=settings.eng_features,
            var_name="label",
        )
        return df_po
    return None


def get_m7_products(df: pd.DataFrame, hourly_trades: pd.DataFrame, date_on: Iterable) -> pd.DataFrame | None:
    """Takes in a latest features dataframe, converts the m7_public_trades_hh, 2h and 4h
    into dataframe format, and creates synthetic features based on those trades. For 2h / 4h
    trades specifically, applies the 'engineer_auction_shaping' strategy to include them as part of the
    final traded price.

    Results are returned in a separate dataframe in long format which can be concatenated with the
    original if desired.
    """
    if df is not None:
        trades_list = []
        for col in df.columns:
            if not df[col].isna().all():
                # extract trades from the json
                raw_trades = speedy_json_to_df(df[col])
                # resample 2h and 4h trades
                resampled_trades = resample_2h_4h_trades(raw_trades, col)
                # get date on for all trades, and traded price
                result = get_m7_trades(resampled_trades, date_on, col)
                trades_list.append(result)
        final_result = pd.concat(trades_list)
        # implement DA auction shaping strategy
        auction_shaped_trades = engineer_auction_shaping(format_trades(final_result, hourly_trades))
        auction_shaped_trades = pd.melt(
            auction_shaped_trades,
            id_vars=["date_on", "date_start"],
            value_vars="traded_price",
            var_name="label",
        ).assign(date_end=lambda x: x["date_start"] + SPL)
        return auction_shaped_trades
    return None


def get_m7_trades(
    df: pd.DataFrame,
    date_on: Iterable,
    market: Literal["m7_public_trades_hh", "m7_public_trades_2h", "m7_public_trades_4h"],
) -> pd.DataFrame:
    """Similar to latest_values_of() in transform.py, this function first parses through each date_on
    and slices the dataframe down to only consider trades placed for the following 12 hours.

    Args:
        df (pd.DataFrame): long data of M7 product (hh, 2h, 4h)
        date_on (Iterable)
        market (str): M7 market (hh, 2h, 4h)

    Returns:
        pd.DataFrame: long data with M7 engineered trades for the specified market
    """
    df = (
        df.assign(tradeExecutionTime=lambda x: pd.to_datetime(x["tradeExecutionTime"]))
        .set_index(["date_start", "tradeExecutionTime"])
        .sort_index()
    )
    engineered_trades = []
    # for each date on, filter out trades that are outside the target lag windows
    for d in date_on:
        start, end = (
            d.floor("30T") + SPL * (settings.min_target_lag - settings.market_lags.get(market)),
            d.floor("30T") + SPL * settings.max_target_lag,
        )
        # 1. filtering for SPs we are predicting for (i.e., 3 to 24 SPs into the future)
        # 2. filtering out any trades that were executed after date_on
        engineered_trades.append(
            get_power_weighted_trades(df.loc[pd.IndexSlice[start:end, :d], :], market).assign(date_on=d)
        )
    return pd.concat(engineered_trades)


def get_power_weighted_trades(df: pd.DataFrame, col: str, threshold_mw: int = 10) -> pd.DataFrame:
    """Gets a traded price for each M7 product (hh, 2h, 4h) when specified in the col arg.
    We then filter based on the total amount of MW traded, and weight the price by volume and
    how long since being traded

    Args:
        df (pd.DataFrame): long data of M7 public trades associated to a specific date_on
        col (str): identifier of which traded product (hh, 2h, 4h)
        threshold (int, optional): minimum traded volume (MW) required to consider "valid market liquidity".
            Defaults to 10.

    Returns:
        pd.DataFrame: long data with one traded price per M7 product specified.
    """
    dfs = []
    for d, dfu in df.groupby("date_start"):
        # Here I want two features from this trade data right now:
        # 1. The number of MW traded
        if dfu["power"].sum() < threshold_mw:
            row = {"date_start": d, "label": col, "traded_price": np.nan}
        else:
            # 2. The price of the the MW traded, weighted by both the volume and how long its been since being traded
            dfu = dfu.reset_index().sort_values(by="tradeExecutionTime", ascending=False).reset_index(drop=True)
            most_recent = dfu["tradeExecutionTime"].max()
            delta = (most_recent - dfu["tradeExecutionTime"]) / np.timedelta64(1, "m")
            decay = (
                most_recent - dfu.loc[dfu["power"].cumsum().ge(threshold_mw).index[0], "tradeExecutionTime"]
            ) / np.timedelta64(1, "m")
            if decay < 10:
                decay = 10
            # weigh prices by multiplying the traded power with a decay rate that halves
            # every timedelta computed as the difference between the most recently executed
            # trade and the 10th MW executed trade time. If this is less than 10 minutes,
            # use 10 (as previous implementation).
            weights = np.abs(dfu["power"] * (0.5 ** (delta / decay)))
            traded_price = np.average(dfu["price"], weights=weights).round(2)
            row = {
                "date_start": d,
                "label": col,
                "traded_price": traded_price,
            }
        dfs.append(row)
    return pd.DataFrame(dfs)


def resample_2h_4h_trades(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """We need to resample the 2h and 4h trades into half hour periods, based on
    the "date_start" column, being the start date of a contract.
    Because there tend to be several trades for one SP start, we cannot resample
    in the traditional way because the index is not unique.

    What therefore has to be done is iterate through each row and multiply out that
    date_start's trades by however long the product is for (2h = 4SP's, 4h = 8SP's.)
    This is a more direct form of resampling, but we don't really have a choice.
    """
    # define how many SP's are in each traded product

    def _resample(trades: pd.DataFrame, market: str) -> pd.DataFrame:
        """
        This does the actual manual resampling. We multiply out the original row by
        the trade length as defined in limit_dict (how many SP's per product).

        For example, for the 2h product it is a list of length 4

        Next, we resample the date_start by multiplying each of those trade lengths
        by 30 minutes, directly resampling the trade for its duration.
        """
        lag = settings.market_lags.get(market)
        return pd.concat([trades] * lag, axis=0, ignore_index=True).assign(
            date_start=pd.concat(
                [trades["date_start"] + i * SPL for i in range(lag)],
                axis=0,
                ignore_index=True,
            ),
            date_end=lambda x: x["date_start"] + SPL,
        )

    return _resample(df, market=market)


def speedy_json_to_df(trades: pd.Series) -> pd.DataFrame:
    """Using orjson, we can quickly extract all of the trades from the nested json in each row
    by loading them into a list and then creating one massive dataframe out of them.
    """
    s = trades.dropna()
    output = None
    indexes = []
    for index, item in s.items():
        item = orjson.loads(item)
        if output is None:
            output = [item]
        else:
            output.append(item)
        indexes += [index] * len(item["data"])
    keys = list(output[0].keys())
    final = defaultdict(list)
    for k in keys:
        for item in output:
            final[k] += item[k]
    df = pd.DataFrame(final["data"], columns=list(dict.fromkeys(final["columns"])))
    df[s.index.name] = indexes
    df = df.assign(label=trades.name).drop(columns="tradeId")
    return df


def format_trades(trades: pd.DataFrame, hourly_trades: pd.DataFrame = None) -> pd.DataFrame:
    """Takes M7 hh, 2h and 4h trades and the hourly auction trades in long format and makes them wide.
    Then adds the respective start dates for 2h and 4h trade periods.

    Args:
        trades (pd.DataFrame): all hh, 2h and 4h trades already resampled by date_on
        hourly_trades (pd.DataFrame, optional): ETS Hourly auction prices already
            resampled by date_on. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    # this is included for the auction shaping strategy
    if hourly_trades is not None:
        df = pd.concat(
            [trades, hourly_trades.drop(columns="date_end").rename(columns={"value": "traded_price"})],
            axis=0,
            ignore_index=True,
        )
    else:
        df = trades.copy()
    # make data long to wide
    df = (
        df.set_index(["date_on", "date_start", "label"])
        .unstack(level="label")
        .droplevel(0, axis=1)
        .reset_index()
        .sort_values(["date_on", "date_start"])
    )
    # get efa and half-efa block start dates into the dataframe
    df[["4h_start", "2h_start"]] = df.apply(parse_efa, result_type="expand", axis=1)
    df = df.rename_axis(None, axis=1).infer_objects()
    return schemas.FormattedTradesDF.validate(df)


def engineer_auction_shaping(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in missing M7 hh trades using a combination of the ets hour auction and 2h/4h trades.
    What we do is:
        1) calculate the average ets price for a certain traded period
        2) find the difference between the ets price per settlement period and the average
        3) apply that difference to the m7 2h/4h prices for that traded period
        4) We then take the average of the 2h/4h shapes prices and apply then to any M7 hh trades that are missing (NaN)

    E.g. for a 2h traded period:
    ets_hour_auction_prices = [100, 100, 120, 120]
    2h_traded_prices = [105, 105, 105, 105]
    ets_hour_average = 110
    shaped_2h = [95, 95, 115, 115]

    Args:
        df (pd.DataFrame): dataframe with hh, 2h, 4h M7 traded prices.

    Returns:
        pd.DataFrame: dataframe with only
    """
    drop_cols = [
        "m7_public_trades_hh",
        "m7_public_trades_2h",
        "m7_public_trades_4h",
        "4h_start",
        "2h_start",
        "shaped_4h",
        "shaped_2h",
        "shaped_avg",
    ]
    # ffill the ets_hour_price by one to ensure it's half-hourly
    df = df.assign(ets_hour_price=lambda x: x["ets_hour_price"].ffill(limit=1))
    dfs = []
    for _, dfu in df.groupby("date_on"):
        dfu = dfu.assign(
            shaped_4h=lambda x: x["m7_public_trades_4h"]
            + (x["ets_hour_price"] - dfu.groupby("4h_start")["ets_hour_price"].transform("mean")),
            shaped_2h=lambda x: x["m7_public_trades_2h"]
            + (x["ets_hour_price"] - dfu.groupby("2h_start")["ets_hour_price"].transform("mean")),
            shaped_avg=lambda x: x[["shaped_2h", "shaped_4h"]].mean(axis=1),
            traded_price=lambda x: x["m7_public_trades_hh"].fillna(x["shaped_avg"]).round(2),
        )
        dfs.append(dfu)
    dfs = pd.concat(dfs).drop(columns=drop_cols)
    return schemas.ShapedTradesDF.validate(dfs)


def engineer_efa_averaging(df: pd.DataFrame) -> pd.DataFrame:
    """Use M7 2h/4h trades to fill in any missing hh trades. This uses the
    "efa_averaging" strategy, which involves finding the average of the hh
    price. The calculation is then 'reverse averaging' the missing hh trades
    to that the average of the hh trades now matches the price for the traded period
    of the 2h/4h product

    Args:
        df (pd.DataFrame): wide dataframe of hh, 2h, and 4h traded prices

    Returns:
        pd.DataFrame: dataframe with one traded price per date_start per date_on
    """
    dfs = []
    drop_cols = [
        "m7_public_trades_hh",
        "m7_public_trades_2h",
        "m7_public_trades_4h",
        "4h_start",
        "2h_start",
    ]
    # iterate through each date_on and efa block start
    for _, dfu in df.groupby(["date_on", "4h_start"]):
        # if there are no missing half hour trades OR there are no 2h/4h traded prices,
        # return the hh as traded price
        if (dfu["m7_public_trades_hh"].count() == len(dfu["m7_public_trades_hh"])) or (
            dfu["m7_public_trades_2h"].count() == 0 & dfu["m7_public_trades_4h"].count() == 0
        ):
            dfu = dfu.assign(traded_price=lambda x: x["m7_public_trades_hh"]).drop(columns=drop_cols)
            dfs.append(dfu)
        else:
            # iterate through each efa block in the date_on (this is because some date_on's are over 2 efa blocks)
            for _, dfu_4h in dfu.groupby("4h_start"):
                # calculate 4h trades average
                avg_4h = dfu_4h["m7_public_trades_4h"].mean()
                # if there are 4h trades and no 2h trades, fill 2h trades in with 4h using
                # reverse averaging
                if not np.isnan(avg_4h) and dfu_4h["m7_public_trades_2h"].isna().to_numpy().any():
                    avg = (len(dfu_4h) * avg_4h - dfu_4h["m7_public_trades_2h"].sum()) / dfu_4h[
                        "m7_public_trades_2h"
                    ].isna().sum()
                    dfu_4h = dfu_4h.assign(m7_public_trades_2h=dfu_4h["m7_public_trades_2h"].fillna(avg))
                # now go through each half efa period
                for _, dfu_2h in dfu_4h.groupby("2h_start"):
                    # if there are no missing hh trades
                    if dfu_2h["m7_public_trades_hh"].count() == len(dfu_2h["m7_public_trades_hh"]):
                        dfu_2h = dfu_2h.assign(traded_price=lambda x: x["m7_public_trades_hh"]).drop(columns=drop_cols)
                        dfs.append(dfu_2h)
                    else:
                        avg_2h = dfu_2h["m7_public_trades_2h"].mean()
                        # reverse average the hh price
                        avg = (len(dfu_2h) * avg_2h - dfu_2h["m7_public_trades_hh"].sum()) / dfu_2h[
                            "m7_public_trades_hh"
                        ].isna().sum()
                        dfu_2h = (
                            dfu_2h.assign(traded_price=dfu_2h["m7_public_trades_hh"])
                            .fillna(avg)
                            .drop(columns=drop_cols)
                        )
                        dfs.append(dfu_2h)
    final = pd.concat(dfs)
    return final


def parse_efa(row: dt) -> tuple[dt, dt]:
    """Gets efa block start dates from dates."""

    london_timezone = ZoneInfo("Europe/London")
    x = row["date_start"]
    hour = x.hour
    date = x.date()
    if hour >= 23:
        date += td(days=1)
    trd_4h_block = 1 + (((hour + 1) % 24) // 4)
    trd_4h_start_hour = (trd_4h_block - 1) * 4 - 1
    trd_2h_block = 1 + (((hour + 1) % 24) // 2)
    trd_2h_start_hour = (trd_2h_block - 1) * 2 - 1
    if trd_4h_block == 1:
        d = date - td(days=1)
        if 1 <= hour < 3:
            trd_2h_start = dt(date.year, date.month, date.day, 1, tzinfo=london_timezone)
            trd_4h_start = dt(d.year, d.month, d.day, 23, tzinfo=london_timezone)
        else:
            trd_2h_start = dt(d.year, d.month, d.day, 23, tzinfo=london_timezone)
            trd_4h_start = dt(d.year, d.month, d.day, 23, tzinfo=london_timezone)
    else:
        trd_4h_start = dt(date.year, date.month, date.day, trd_4h_start_hour, tzinfo=london_timezone)
        trd_2h_start = dt(date.year, date.month, date.day, trd_2h_start_hour, tzinfo=london_timezone)
    return trd_4h_start, trd_2h_start


def encode_dates(df: pd.DataFrame) -> pd.DataFrame:
    _, df["sp"] = period.series_from_times(df.date_start)
    df["sp_sin"] = df["sp"].apply(lambda x: np.sin(x * 2 * np.pi / 48))
    df["sp_cos"] = df["sp"].apply(lambda x: np.cos(x * 2 * np.pi / 48))
    return df.drop("sp", axis=1)


def ensure_has_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    # TODO Q1 2024 HERE COMES PANDERAAAAA
    """Ensures dataframe includes the given columns

    Args:
        df (pd.Dataframe): the dataframe with the features
        columns (list[str]): the columns we need to ensure that are included

    Returns:
        pd.Dataframe: the dataframe including the columns even with nan values
    """
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
            df[col] = df[col].ffill()
    return df


def latest_lolp(df: pd.DataFrame) -> pd.DataFrame:
    """Specifically for the lolp feature, it returns the latest one and renames it to simply lolp

    Args:
        df (pd.DataFrame): The df with the features

    Returns:
        pd.DataFrame: The updated df
    """
    # select one lolp only and rename it to lolp only
    lolp_cols = df.columns[df.columns.str.contains("lolp")]
    if len(lolp_cols) > 0:
        # TODO: The below has a bunch of assumptions about the column name
        # TODO: Which we could remove, and also this feels like a function that
        # TODO: might be useful for other models, so maybe we should add
        # TODO: a trans.latest_value_from_cols or similar
        min_lolp = min([int(x.split("_")[1].split("hour")[0]) for x in lolp_cols])
        df = df.drop(lolp_cols[lolp_cols != f"lolp_{min_lolp}hour"], axis=1)
        df = df.rename(columns={f"lolp_{min_lolp}hour": "lolp"})
        df["lolp"] = df["lolp"].fillna(0)
    else:
        df["lolp"] = 0
    return df


def natural_keys(text, split=re.compile(r"(\d+)")):
    return [int(s) if s.isdigit() else s.lower() for s in split.split(text)]


def get_raw_data(fs: FeatureStore, start: dt, end: dt):
    features = settings.all_features_incl_target
    latest_features = settings.latest_features

    # @cache(Path(__file__).parent.parent / "cache")
    def _download_features(features, latest_features, start, end):
        return (
            fs.get_long_features(features, start, end, chunk_timedelta=td(days=14)).drop(columns="date_retrieved"),
            fs.get_latest_features(latest_features, start, end, chunk_timedelta=td(days=14)),
        )

    return _download_features(features, latest_features, start, end)
