import json
import re
from datetime import timedelta as td

import numpy as np
import pandas as pd


def natural_keys(text, split=re.compile(r"(\d+)")):
    return [int(s) if s.isdigit() else s.lower() for s in split.split(text)]


def check_tos_distance(row):
    """Check the distance between the top buy and sell in the order stack and
    replace huge differences with the decaying traded price.
    """
    # TODO improve this logic. The 2x factor is a bit arbitrary.
    # Although very rare to see, it probably does not work with huge prices i.e buy of 800 and sell of 2000.
    distance = row["weighted_top_sell"] - row["weighted_top_buy"]
    if (distance > 2 * np.abs(row["weighted_top_buy"])) & (distance > 10):
        row["weighted_top_sell"] = row["traded_price"]
        row["weighted_top_buy"] = row["traded_price"]
    return row


def handle_missing_data_features(df):
    """
    Fill nan values in the most meaningful way, instead of simple ffill/bfill
    """
    if "ets_hour_price" in df.columns:
        df = df.drop(["ets_hour_price"], axis=1)
    if "ets_half_hour_price" in df.columns:
        df = df.drop(["ets_half_hour_price"], axis=1)
    if "traded_mw" in df.columns:
        df = df.drop(["traded_mw"], axis=1)
    if "weighted_top_buy" in df.columns and "weighted_top_sell" in df.columns:
        df["weighted_top_buy"] = df["weighted_top_buy"].fillna(df["traded_price"])
        df["weighted_top_sell"] = df["weighted_top_sell"].fillna(df["traded_price"])
        df = df.apply(check_tos_distance, axis=1)

    # If there are NaNs in first row forward method leaves them as they were
    # We are using bfill after the forward fill to ensure that the oldest
    # instances are filled with values. Rarely more than the 3-4 first rows
    return df.ffill().bfill()


# TODO I don't think we'll ever use the following features ever again
# so probably worth removing them at some point.
def extract_m7(x):
    if isinstance(x, float) and np.isnan(x):
        return x
    x = json.loads(x)
    x = pd.DataFrame(x["data"], columns=x["columns"])
    x["tradeExecutionTime"] = pd.to_datetime(x["tradeExecutionTime"])
    x = x.sort_values(["tradeExecutionTime", "tradeId"], ascending=False).reset_index(drop=True)
    return x


def get_ema_price(df: pd.DataFrame, mins: int = 5):
    # Yes I am checking for NaNs. I will use a better name for df
    if isinstance(df, float) and np.isnan(df):
        return df
    else:
        return (
            df["price"]
            .ewm(
                halflife=td(minutes=mins),
                times=df["tradeExecutionTime"],
            )
            .mean()
            .to_numpy()[-1]
        )  # As we sorted by time when making the df


def get_macd(df: pd.DataFrame):
    ema12 = get_ema_price(df, mins=6)
    ema26 = get_ema_price(df, mins=13)
    return ema12 - ema26


def get_atr(df: pd.DataFrame):
    if isinstance(df, float) and np.isnan(df):
        return df
    else:
        return abs(df["price"].max() - df["price"].min())


def technical_indicators(df: pd.DataFrame, eng_feats: list[str]):
    if "m7_public_trades_hh" in df.columns:
        # convert json on each row into pd.DataFrames - required for all functions below
        df["m7_public_trades_hh"] = df["m7_public_trades_hh"].apply(lambda x: extract_m7(x))
        # each cell of m7_public_trades_hh is now a DataFrame with all trades ordered by exec time
        # if "ema_half_hour" in eng_feats:
        # df["ema_half_hour"] = df["m7_public_trades_hh"].apply(lambda df: get_ema_price(df))
        if "MACD" in eng_feats:
            df["MACD"] = df["m7_public_trades_hh"].apply(lambda df: get_macd(df))
        if "atr" in eng_feats:
            df["atr"] = df["m7_public_trades_hh"].apply(lambda df: get_atr(df))
    else:
        tech_ind_cols = ["MACD", "atr"]
        df = df.join(
            pd.DataFrame(
                np.nan,
                index=df.index,
                columns=[col for col in tech_ind_cols if col in eng_feats],
            )
        )
    return df
