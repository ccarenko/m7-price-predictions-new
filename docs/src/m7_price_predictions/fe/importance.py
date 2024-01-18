from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from m7_price_predictions.fe.transform import get_scaled_data
from m7_price_predictions.m7_settings import settings
from xgboost import XGBRegressor


def plot_feature_importances(model, feature_names, save_loc, sp, num_feats=20):
    """Make a matplotlib plot of the feature importances and save to file"""
    fig, ax = plt.subplots(figsize=(8, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)
    ax.set_title("XGBoost Feature importances")
    ax.barh(feature_names[indices][:num_feats], importances[indices][:num_feats])
    fig.tight_layout()
    fig.savefig(save_loc)


def get_data(sp=3):
    start = datetime(2022, 6, 1, tzinfo=pytz.utc)
    end = datetime(2023, 1, 5, tzinfo=pytz.utc)
    # check numer of target lag returned, it might not make sense to tune over all of them
    cache = get_scaled_data(start, end, settings)

    # select only 3 time horizons to test - np.array([3, 6, 9])
    # cache["X"] = [cache["X"][3]] + [cache["X"][6]] + [cache["X"][9]]
    # cache["y"] = [cache["y"][3]] + [cache["y"][6]] + [cache["y"][9]]
    # cache["target_lags"] = np.array([3, 6, 9])
    data = {}
    new_engineered_features = ["traded_price_2h", "traded_price_4h", "traded_price_hh"]
    cols = settings.features + settings.engineered_features + ["lolp"] + new_engineered_features
    cols = [
        c
        for c in cols
        if c
        not in [
            "m7_orderbook_hh",
            "m7_public_trades_hh",
            "m7_public_trades_2h",
            "m7_public_trades_4h",
            "lolp_2hour",
            "lolp_4hour",
            "lolp_8hour",
            "lolp_12hour",
        ]
    ]
    data["X"] = pd.DataFrame(cache["X"], columns=cols)
    data["y"] = pd.DataFrame(cache["y"], columns=[settings.target])
    return data


if __name__ == "__main__":
    SP = 9
    data = get_data(SP)
    X, y = data["X"], data["y"]

    xgb = XGBRegressor(n_estimators=200)
    xgb.fit(X, y)

    doc_dir = Path(__file__).parent.parent.parent / "docs"

    plot_feature_importances(xgb, X.columns, doc_dir / "xgboost_importance_test.png", sp=SP)
