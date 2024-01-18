import os
import platform
from datetime import datetime as dt
from datetime import timedelta as td
from datetime import timezone as tz
from pathlib import Path

import arenkodsml
import arenkodsml.data.fs.scaling
import cloudpickle
import m7_price_predictions
import mlflow
import numpy as np
import tensorflow as tf
from arenkodsml.common.utils import better_describe
from arenkodsml.environments import Environment, get_environments_from_env_key
from arenkodsml.serving.quantile_keras import QuantileKeras
from loguru import logger
from m7_price_predictions.fe.transform import complex_transform, get_data, inverse_scale_target, scaling_locally
from m7_price_predictions.m7_settings import settings
from m7_price_predictions.models.quantile_neural_network import load_models, m7_baseline, save_models
from m7_price_predictions.tuning.tuning import get_params
from m7_price_predictions.utils.alerting import alert

cloudpickle.register_pickle_by_value(arenkodsml.data.fs.scaling)
cloudpickle.register_pickle_by_value(arenkodsml)
cloudpickle.register_pickle_by_value(m7_price_predictions)

if "arm" in platform.machine():
    tf.config.set_visible_devices([], "GPU")


@alert.on_exception(reraise=True)
def train_m7(date_start: dt, date_end: dt, load_predictors: bool = False, local: bool = False) -> None:
    # ensure right inputs if running on the cloud for automatic model retraining
    if not local:
        load_predictors = False

    run_name, exp_name = "NNs", "m7_gate_close_price"
    logger.info(f"Training {exp_name} from {date_start} to {date_end}")

    # test connection to envs
    envs = get_environments_from_env_key(run_name, exp_name)
    for e in envs:
        e.validate_connection()

    # constants
    quantiles = settings.quantiles

    logger.info("Fetching data")
    unscaled_data = get_data(date_start, date_end)
    df = unscaled_data["X"].copy()
    if settings.yeo_scale_target:
        df[settings.target] = unscaled_data["y"]
    scaling_df = better_describe(df)
    logger.info("Scaling data")
    output = scaling_locally(scaling_df, unscaled_data.copy())

    # load optimal params
    params = get_params()

    # add extra kwargs
    params.update(
        num_features=output["X"].shape[1],
        output_size=quantiles.size,
        quantiles=quantiles,
    )

    logger.info("Initialising Keras model")
    keras_model = QuantileKeras(
        target=settings.target,
        transformation_func=complex_transform,
        scaling_df=scaling_df,
        target_lags=np.array([0]),
        quantiles=quantiles,
        invert_scale_target=inverse_scale_target,
    )

    path_dir = Path(__file__).parent / "trained_models"

    def train_model(model, X, y):
        model.fit(X, y)
        return model

    logger.info("Creating keras graph and loss")
    baseline_model = m7_baseline(**params)
    custom_loss = baseline_model.get_loss()

    if load_predictors and path_dir.exists():
        baseline_model.model = load_models(custom_loss, path_dir)
    else:
        logger.info("Fitting keras model")
        baseline_model = train_model(baseline_model, output["X"], output["y"])

        save_models(baseline_model, path=path_dir)

    # set models to wrapper and save out custom_loss
    keras_model.set_model(baseline_model.get_model())

    tags = {
        "train_start": date_start.isoformat(),
        "train_end": date_end.isoformat(),
    }

    # data trained on
    params["date_train_start"] = date_start.isoformat()
    params["date_train_end"] = date_end.isoformat()
    name = "m7_price"

    def deploy(env: Environment, stage: str) -> None:
        mlflow.log_params(params)
        env.log_model(
            keras_model,
            name,
            stage,
            custom_objects=custom_loss,
            extra_tags=tags,
        )

    # deploy on all env inside env_list and register params and tags
    for env in envs:
        with env as e:
            deploy(e, "staging")


if __name__ == "__main__":
    # os.environ["DS_FS_EXCHANGE_ADDR"] = "http://172.22.11.26:5903" # dat3
    os.environ["DS_FS_EXCHANGE_ADDR"] = "http://172.29.11.20:5903"  # prod
    # training dates
    date_end = (dt.now(tz=tz.utc) - td(hours=24)).replace(hour=0, minute=0, second=0, microsecond=0)
    date_start = date_end - td(days=365)

    # We really need to standardise how all our predictors look
    local = False
    if os.environ.get("ML_ENVS") is None:
        os.environ["ML_ENVS"] = "dat3" if local else "dat3,prod"

    # run the main scripts
    train_m7(date_start, date_end, load_predictors=False, local=local)
