""" Neural network w/ quantile loss """
import logging

import tensorflow as tf
from arenkodsml.baseline_estimators import QuantileNeuralNetwork


def m7_baseline(
    num_features: int,
    num_layers: int,
    num_neurons: int,
    dropout: float,
    output_size: int,
    num_timesteps: int = 1,
    **params,
):
    """Generate baseline model"""
    graph = tf.keras.Sequential()
    graph.add(tf.keras.layers.Input(shape=(num_features,)))
    for i in range(num_layers):
        graph.add(
            tf.keras.layers.Dense(
                num_neurons,
                kernel_initializer="he_uniform",
                bias_initializer=tf.keras.initializers.Zeros(),
                name=f"layer_{i}",
            )
        )
        graph.add(tf.keras.layers.LeakyReLU(alpha=0.1, name=f"leaky_relu_{i}"))
        graph.add(
            tf.keras.layers.Dropout(dropout, name=f"dropout_{i}"),
        )
    graph.add(tf.keras.layers.Dense(output_size))
    graph.add(tf.keras.layers.Reshape((num_timesteps, output_size)))
    params["verbose"] = 0
    logging.info(f"Model architecture: {num_neurons}x{num_layers}, {dropout=}")
    return QuantileNeuralNetwork(graph=graph, num_timesteps=num_timesteps, **params)


def save_models(model, path):
    model.model.save(path)


def load_models(loss, path):
    model = tf.keras.models.load_model(path, custom_objects={"loss": loss})
    return model
