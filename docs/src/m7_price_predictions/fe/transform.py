from datetime import datetime as dt
from datetime import timedelta as td
from datetime import timezone as tz
from pathlib import Path

import arenkods.data.fs.transformations as tran
import numpy as np
import pandas as pd
from arenkods.common.utils import cache
from arenkods.data.fs import FeatureStore
from arenkods.interval.constant import SETTLEMENT_PERIOD_LENGTH as SPL
from arenkodsml.common.utils import better_describe
from arenkodsml.data.fs.scaling import Scaler, scaling_funcs
from m7_price_predictions.fe.features import add_engineered_features, encode_dates, ensure_has_columns, get_raw_data
from m7_price_predictions.fe.utils import handle_missing_data_features
from m7_price_predictions.m7_settings import M7Settings, settings


def get_target_df(response: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    From the response dataframe, collect the target label. Sort and order dataframe for use
    """

    targets_df = (
        response.loc[response["label"] == target, ["date_start", "value"]]
        .set_index("date_start")
        .rename(columns={"value": target})
        .sort_index()
        .ffill()
        .bfill()
    )
    return targets_df


def prepare_features_list(
    raw_data: pd.DataFrame,
    trades_data: pd.DataFrame,
    start: dt,
    end: dt,
    scaling_df: pd.DataFrame = None,
):
    time_horizons = settings.time_horizons
    df_list = []

    date_on = pd.date_range(start, end - SPL, freq="30T", tz=tz.utc).floor("30T") + td(minutes=21)

    # select all features except target and get latest values of according to date_on
    data = (
        raw_data[~raw_data["label"].str.contains("m7_gate_price_mean")]
        .pipe(tran.latest_values_as_of, date_on, threshold=SPL * (5 + settings.max_threshold))
        .drop(columns="date_published")
    )
    df = (
        add_engineered_features(data, trades_data, date_on)
        .assign(
            step_length=lambda x: x["date_end"] - x["date_start"],
            upsample=lambda x: x["step_length"] > SPL,
            date_end=lambda x: x["date_start"] + SPL,
        )
        .drop(columns=["step_length", "upsample"])
        .reset_index()
        .sort_values(by=["date_on", "date_start"])
        .pivot(
            index=["date_on", "date_start", "date_end"],
            columns=["label"],
            values=["value"],
        )
        .droplevel(0, axis=1)
        .reset_index()
        .assign(ets_hour_price=lambda x: x["ets_hour_price"].ffill())
    )

    if "ets_half_hour_price" in df.columns:
        df["ets_half_hour_price"] = df["ets_half_hour_price"].fillna(df["ets_hour_price"])
    else:
        df["ets_half_hour_price"] = df["ets_hour_price"]

    # calculate price delta when traded_price is available, then ffill
    # last available delta using exponential decay. Use newly calculated
    # deltas to to fill missing traded_price rows.
    decay_factors = (0.8) ** (df["traded_price"].groupby(df["traded_price"].notna().cumsum()).cumcount() - 1)
    df = (
        df.assign(
            price_delta=lambda x: x["ets_half_hour_price"] - x["traded_price"],
            price_delta_ffilled=lambda x: x["price_delta"].ffill(),
        )
        .assign(
            price_delta=lambda x: x["price_delta"].fillna(x["price_delta_ffilled"].mul(decay_factors, 0)),
            traded_price=lambda x: x["traded_price"].fillna(x["ets_half_hour_price"] - x["price_delta"]).round(2),
        )
        .drop(columns=["price_delta", "price_delta_ffilled"])
    )
    data = pd.melt(
        df,
        id_vars=["date_on", "date_start", "date_end"],
        value_vars=df.columns,
        var_name="label",
        value_name="value",
    )[["date_on", "label", "date_start", "date_end", "value"]]

    for time in time_horizons:
        # set date_start according to timing of predictions
        date_start = date_on.floor("30T") + time

        # find the latest forecasts at each date_on according to the time horizon
        features_df = (
            tran.long_to_wide(data, date_on=date_on, date_start=date_start)
            .infer_objects()
            .reset_index(drop=True)
            .assign(date_on=lambda x: x["date_on"].dt.floor("30T"))
            .merge(
                pd.DataFrame(index=pd.date_range(start, end - td(minutes=30), freq="30min")),
                how="right",
                left_on=["date_on"],
                right_index=True,
            )
            .assign(date_start=lambda x: x["date_on"] + time)
            .pipe(encode_dates)
            .loc[lambda x: (x["date_on"] >= start) & (x["date_on"] <= end)]
            .set_index("date_on")
            .loc[lambda x: ~x.index.duplicated(keep="first")]
            .asfreq(freq="30T")
        )

        # add artificial columns with np.nan
        if scaling_df is not None:
            features = [x for x in scaling_df.index.tolist() if x != settings.target]
            features_df = ensure_has_columns(features_df, features)
        elif len(df_list) > 0:
            features_df = ensure_has_columns(features_df, df_list[0].columns)

        df_list.append(
            features_df.replace([np.inf, -np.inf], np.nan)
            .pipe(handle_missing_data_features)
            .drop(columns="date_on", errors="ignore")
            .reset_index()
        )

    df_all = pd.concat(df_list, axis=0, ignore_index=True).assign(
        sp_delta=lambda x: (x["date_start"] - x["date_on"]) / SPL
    )

    return df_all


def scaling_locally(scaling_df: pd.DataFrame, output: dict) -> dict:
    """Scale data locally so training set does not have to go through the transform again"""
    features_df = output["X"]
    # make sure we have all features
    missing = [f for f in scaling_df.index if f not in features_df.columns]
    for m in missing:
        features_df[m] = scaling_df.loc[m, "50%"]

    nan_cols = features_df.columns[features_df.isna().any()].tolist()
    if nan_cols:
        for c in nan_cols:
            features_df[c] = scaling_df.loc[c, "50%"]

    # ensure same order as in training (mostly for production situations)
    features_df = features_df[scaling_df.index]

    # scale and replace
    features_df = scale_features(features_df, scaling_df=scaling_df)

    # convert to numpy array and save
    output["X"] = features_df.fillna(0)

    if "y" in output:
        t = settings.target
        y = output["y"]
        y = pd.DataFrame(
            y,
            columns=[t],
            index=output["index"],
        )
        y = scale_target(y, scaling_df)
        output["y"] = y.to_numpy().astype(np.float64)

    return output


def scale_features(X: pd.DataFrame, scaling_df: pd.DataFrame) -> pd.DataFrame:
    """Scale features"""
    scaler = Scaler(X, scaling_df, check_all_cols_scaled=True)
    if settings.yeo_scale_prices:
        scaler.add_rule(".*price.*", scaling_funcs.yeo_johnson_scaling)
    scaler.add_rule(".*", scaling_funcs.quantile_range_scaling)
    df_scaled = scaler.apply()
    return df_scaled


def scale_target(y: pd.DataFrame, scaling_df: pd.DataFrame) -> pd.DataFrame:
    if not settings.yeo_scale_target:
        return y
    scaler = Scaler(y, scaling_df, check_all_cols_scaled=True)
    scaler.add_rule(".*", scaling_funcs.yeo_johnson_scaling, source=settings.target)
    return scaler.apply()


def inverse_scale_target(
    scaling_df: pd.DataFrame,
    y_scl: pd.DataFrame,
) -> pd.DataFrame:
    """Return inverse-transformed input following Yeo-Johnson inverse
    transform with parameter lambda.
    """
    if y_scl is None:
        return None
    scale = settings.yeo_scale_target
    if not scale:
        return y_scl
    scaler = Scaler(y_scl, scaling_df, check_all_cols_scaled=False)
    scaler.add_rule(r"0\..*", scaling_funcs.yeo_johnson_invert, source=settings.target)
    return scaler.apply()


def complex_transform(
    fs: FeatureStore,
    start: dt,
    end: dt,
    scaling_df: pd.DataFrame | None = None,
    cache: dict | None = None,
) -> dict:
    """Transform function for the gate_close price on M7 - multiple NN predictions"""

    output = None
    if cache is not None:
        output = cache.copy()
    else:
        max_lag = SPL * (5 + settings.max_past_lag)
        end_time = end + SPL * (settings.max_target_lag + 1)
        raw_data, trade_features = get_raw_data(fs, start - max_lag, end_time + max_lag)

        targets_indexed = get_target_df(raw_data, settings.target)

        X = prepare_features_list(raw_data, trade_features, start, end, scaling_df=scaling_df)

        y = X.merge(targets_indexed, on="date_start", how="left")[settings.target].interpolate().ffill().bfill()

        output = {
            "date_on": X["date_on"].to_numpy(),
            "date_start": X["date_start"].to_numpy(),
            "index": X["date_start"].to_numpy(),
            "X": X.drop(columns=["date_on", "date_start"]),
            "y": np.array(y).T.reshape(-1, 1),
            "target_lags": settings.target_lags,
        }
    # Revisit scaling later
    if output is not None and scaling_df is not None:
        output = scaling_locally(scaling_df, output)

    return output


@cache(Path(__file__).parent.parent / "cache")
def get_cached_transform(start: dt, end: dt) -> dict:
    """Wrapper function to quickly get a simple test data set
    for playing with simple models. By keeping the simple transform
    as basic as possible, we can pull large amounts of data easily."""
    return complex_transform(FeatureStore(), start, end)


def get_data(start: dt, end: dt) -> dict:
    return get_cached_transform(start, end)


@cache(Path(__file__).parent.parent / "cache")
def get_cached_transform_with_settings(start: dt, end: dt, settings: M7Settings):
    """Wrapper function to quickly get a simple test data set
    for playing with simple models. By keeping the simple transform
    as basic as possible, we can pull large amounts of data easily."""
    return complex_transform(FeatureStore(), start, end)


def get_scaled_data(start, end, settings: M7Settings):
    output = get_cached_transform_with_settings(start, end, settings)
    scaling_df = better_describe(output["X"])
    return scaling_locally(scaling_df, output.copy())


if __name__ == "__main__":
    import os

    address = "172.22.11.26"
    os.environ["DS_FS_EXCHANGE_ADDR"] = f"http://{address}:5903"

    date_start = dt(2022, 4, 4, 0, 0, tzinfo=tz.utc)
    date_end = dt(2022, 4, 5, 23, 0, tzinfo=tz.utc)

    output = complex_transform(FeatureStore(), date_start, date_end)
