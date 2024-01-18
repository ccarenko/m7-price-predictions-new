import json
import os
import platform
from datetime import datetime as dt
from datetime import timedelta as td
from datetime import timezone as tz
from pathlib import Path

import plotly.express as px
import tensorflow as tf
from arenkods.common.utils import cache
from arenkodsml.tuning.hyperparameter_optimsation import ParameterOptimiser
from arenkodsml.validation.cross_validation import RollingWindowValidation
from arenkodsml.validation.objective import average_quantile_loss
from hyperopt import hp, space_eval
from loguru import logger
from m7_price_predictions.fe.transform import get_scaled_data
from m7_price_predictions.m7_settings import M7Settings, settings
from m7_price_predictions.models.quantile_neural_network import m7_baseline

if "arm" in platform.machine():
    tf.config.set_visible_devices([], "GPU")
os.environ["DS_FS_EXCHANGE_ADDR"] = "http://172.22.11.26:5903"


def get_params() -> dict[str, float]:
    params_file = Path(__file__).parent / "docs" / "best_params.json"
    if params_file.is_file():
        with Path.open(params_file) as f:
            params = json.load(f)
    else:
        params = settings.default_hyperparams
    params.update(
        shuffle=False,
        verbose=1,
    )

    return params


def tune():
    settings = M7Settings(tuning=True)

    # data set to tune on
    end = dt(2023, 1, 2, 0, 0, tzinfo=tz.utc)
    start = end - td(days=365)

    exp_name = "m7_gate_close_price"
    logger.info(f"Training {exp_name} from {start} to {end}")
    # check numer of target lag returned, it might not make sense to tune over all of them
    save_dir = Path(__file__).parent / ".cache"

    @cache(save_dir)
    def get_data(start, end):
        return get_scaled_data(start, end, settings)

    logger.info("Loading data")
    cached_data = get_data(start, end)
    logger.info("Data loaded")
    quantiles = settings.quantiles
    loss_func = average_quantile_loss(quantiles)
    parent = Path(__file__).parent / "docs"
    space = {
        "num_layers": hp.choice("hidden_layers", [1, 2, 3, 4, 5]),
        "num_neurons": hp.uniformint("neurons", 50, 500),
        "dropout": hp.uniform("dropout", 0.0, 0.5),
        "batch_size": hp.choice("batch_size", [1024]),
        "epochs": hp.choice("epochs", [100, 200, 250, 300, 350]),
    }

    # confirm these values
    rolling_window = RollingWindowValidation(train_duration=td(days=310), test_duration=td(days=10))

    parameters = {
        "quantiles": quantiles,
        "num_features": cached_data["X"].shape[1],
        "output_size": quantiles.size,
        "shuffle": False,
        "verbose": 0,
    }

    cached_data["X"] = cached_data["X"].to_numpy()

    objective = rolling_window.objective(
        None,
        None,
        None,
        loss_func=loss_func,
        baseline_model=m7_baseline,
        serving_model=None,
        cache=cached_data,
        parameters=parameters,  # to initialise the baseline model
        # settings=settings, # no need as transform function is not going ot be used
        num_jobs=1,
    )
    num_samples = 50
    optimiser = ParameterOptimiser(
        objective,
        config=space,
        filename=parent / "trials_tuning.pkl",
        use_cache=True,
        max_evals=num_samples,
    )
    logger.info("Starting optimisation")
    optimiser.optimise()

    # print best hyperparameters
    best = space_eval(space, optimiser.trials.argmin)

    # print best hyperparameters
    for k, v in best.items():
        logger.info(f"=> {k}: {v}")

    # save out best results
    with Path.open(parent / "best_params.json", "w") as f:
        json.dump(best, f)

    # make plots
    optimiser.plot_convergence(parent / "convergence_plot.png")
    optimiser.plot_parallel_coordinates(
        parent / "parallel_coordinates_plot.png",
        num_samples=num_samples,
        color="loss",
        color_continuous_scale=px.colors.sequential.Viridis,
    )


if __name__ == "__main__":
    tune()
