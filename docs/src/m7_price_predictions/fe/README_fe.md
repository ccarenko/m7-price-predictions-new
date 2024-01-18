## Input features

<br />

After assessing the results from a combination of local experiments the following features were chosen for the modelling:

            "ets_hour_price",
            "ets_half_hour_price",
            "m7_orderbook_hh",
            "m7_public_trades_hh",

<br />

- The `ets_hour_price` is only used to fill missing values in the `ets_half_hour_price` after a simple resampling step of the former. The hourly price is then dropped and not used as an input feature.

- From the `m7_orderbook_hh` we extract the `pressure` features (i.e., a measure of market inertia) as well as `weighted_top_buy` and `weighted_top_sell`. For example, the `100 pressure` indicates how many MW do we have to buy from the top of the stack until the price changes by £100. These features are for both the buy and sell sides.

- The last feature is processed using the `extract_m7_trade_features` function in `features.py`, which takes in a long-format dataframe, converts the `m7_public_trades_hh`
into dataframe format, and creates synthetic features based on those trades. Results are returned in a separate dataframe in long format which can be concatenated with the original if desired.
This process returns the main feature of interest `traded_price`.
Given a certain settlement period, this feature might not be available for all the time horizons we want to predict. For this reason, we fill the gaps by first computing the difference between the last available `traded_price` and `ets_half_hour_price` and decaying this gap until a new `traded_price` becomes available again.
We have found that filling the `traded_price` following this criterion results in a higher correlation coefficient with the target, especially between 2 and 6 hours ahead. The following plot shows the correlation coefficient of the new engineered feature as opposed to simply filling missing prices with the half hourly price (here "benchmark").
<br />

<details>
<img src="src/m7_price_predictions/eda/traded_price_decay.png" width="1000">
<br />

<br />
As part of a standard procedure followed for other predictors as well, we also add a feature to sine/cosine encode the hour of the day.
In conclusion, the full set of input features is as follows:

            "ets_half_hour_price"
            "traded_price"
            "sp_sin"
            "sp_cos"
            "weighted_top_buy"
            "weighted_top_sell"
            "-10 pressure"
            "-100 pressure"
            "-1000 pressure"
            "10 pressure"
            "100 pressure"
            "1000 pressure"

<br />

Initially, we introduced two additional engineered features:
- `atr` (Average True Range), a measure of market volatility computed as the difference between the maximum and minimum price.
- `MACD` (Moving Average Convergence Divergence). Computed as `ema12-ema26`, which are the exponential moving averages taking into account the last 12 and 26 trades respectively.

However, these features had a negative impact on the predictor, which was returning large troughs when predicting further in the future (see plots attached).

<br />

<details>
<img src="src/m7_price_predictions/eda/troughs_w_eng_feats.png" width="1000">
<img src="src/m7_price_predictions/eda/new_model_wout_eng_feats.png" width="1000">
<br />
