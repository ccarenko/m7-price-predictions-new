from datetime import timedelta as td
from typing import ClassVar

import numpy as np
from arenkodsml.quantiles import get_quantiles


class M7Settings:
    minutes_before_sp: int = 10
    target: str = "m7_gate_price_mean"
    yeo_scale_target: bool = False
    yeo_scale_prices: bool = False
    features: ClassVar[list[str]] = [
        "m7_orderbook_hh",
        "ets_hour_price",
        "ets_half_hour_price",
    ]
    latest_features: ClassVar[list[str]] = [
        "m7_public_trades_hh",
        "m7_public_trades_2h",
        "m7_public_trades_4h",
    ]
    eng_features: ClassVar[list[str]] = [
        "-10 pressure",
        "-100 pressure",
        "-1000 pressure",
        "10 pressure",
        "100 pressure",
        "1000 pressure",
        "weighted_top_sell",
        "weighted_top_buy",
    ]
    max_past_lag: int = 0
    max_target_lag: int = 24  # Inclusive
    min_target_lag: int = 3
    quantiles: np.ndarray = get_quantiles()
    threshold: td = td(hours=14)  # Hours before start time we are getting data for
    default_hyperparams: ClassVar[dict[str, float]] = {
        "epochs": 200,
        "dropout": 0.06,
        "batch_size": 1024,
        "num_layers": 3,
        "num_neurons": 150,
    }
    market_lags: ClassVar[dict[str, int]] = {
        "m7_public_trades_hh": 1,
        "m7_public_trades_2h": 4,
        "m7_public_trades_4h": 8,
    }

    def __init__(self, tuning: bool = False):
        self.tuning = tuning

    @property
    def all_features_incl_target(self):
        return [*self.all_features, self.target]

    @property
    def all_features(self):
        return self.features

    @property
    def engineered_features(self):
        return self.eng_features

    @property
    def past_lags(self):
        return np.arange(self.max_past_lag).astype(int)

    @property
    def target_lags(self):
        if self.tuning:
            return np.linspace(self.min_target_lag, self.max_target_lag, 4).astype(int)
        else:
            return np.arange(self.min_target_lag, self.max_target_lag + 1).astype(int)

    @property
    def all_lags(self):
        return np.arange(-self.max_target_lag, self.max_past_lag + 1).astype(int)

    @property
    def max_threshold(self):
        return max(abs(self.max_past_lag), abs(self.max_target_lag))

    @property
    def time_horizons(self):
        return [td(minutes=30 * float(m)) for m in self.target_lags]


settings = M7Settings()
