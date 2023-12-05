import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from random import random

import uvicorn
from definitions.models.example import ExampleSimple
from definitions.pulsar_client import Producer, PulsarClient
from fastapi import FastAPI
from opentelemetry.trace import Tracer

from pulsar_tracing.prom import PrometheusMiddleware, metrics
from pulsar_tracing.settings import Settings
from pulsar_tracing.tracing import add_tracing, trace_reader, tracer


class Application(FastAPI):
    def __init__(self, lifespan):
        super().__init__(lifespan=lifespan)

    def init(self, settings: Settings, producer: Producer, tracer: Tracer) -> None:
        self.settings = settings
        self.producer = producer
        self.tracer = tracer


@asynccontextmanager
async def lifespan(app: Application) -> AsyncGenerator[None, None]:
    settings = Settings()
    pulsar_client = PulsarClient(settings.pulsar_host)

    # Create the producer we need to emit market price options
    logging.info(f"Creating producer for {settings.full_topic_producer}")
    producer = pulsar_client.create_producer(
        settings.full_topic_producer,
        ExampleSimple,
        timeout_seconds=1.0,
    )
    app.init(settings, producer, tracer)
    logging.info(f"Creating consumer for {settings.full_topic_consumer}")
    coro = pulsar_client.register_reader(
        settings.full_topic_consumer,
        ExampleSimple,
        async_callback=consume_example,
    )
    coroutines = [coro]
    futures = asyncio.gather(*coroutines)
    yield
    logging.error(f"Closing down server {settings.app_name}")
    futures.cancel()
    await asyncio.sleep(0.1)
    producer.close()


@trace_reader
async def consume_example(msg: ExampleSimple | None) -> None:
    if msg is None:
        return
    logging.info(f"Consuming {msg} from {app.settings.pulsar_topic_consume}")
    await asyncio.sleep(0.3 * random())
    msg2 = add_tracing(ExampleSimple(msg=msg.msg, num=msg.num * 10))
    await app.producer.send(msg2)
    logging.info(f"Sending {msg2} to {app.settings.pulsar_topic_produce}")


init_settings = Settings()
time.sleep(init_settings.sleep)
app = Application(lifespan=lifespan)
app.add_middleware(PrometheusMiddleware, app_name=init_settings.app_name)
app.add_route("/metrics", metrics)

if __name__ == "__main__":
    from uvicorn.config import LOGGING_CONFIG

    log_format = (
        "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] "
        "[trace_id=%(otelTraceID)s span_id=%(otelSpanID)s resource.service.name=%(otelServiceName)s] - %(message)s"
    )
    log_config = LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = log_format
    logging.basicConfig(level=logging.INFO, format=log_format)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=log_config)
