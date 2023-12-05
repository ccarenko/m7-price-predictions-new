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
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, Tracer, set_span_in_context

from pulsar_tracing.settings import Settings
from pulsar_tracing.tracing import get_tracer


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
    tracer = get_tracer(settings.app_name, settings.tempo_endpoint)
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


async def consume_example(msg: ExampleSimple | None) -> None:
    if msg is None:
        logging.error("Received None message")
        return
    span_context = SpanContext(
        trace_id=int(msg._properties["trace_id"]),
        span_id=int(msg._properties["span_id"]),
        is_remote=True,
        trace_flags=TraceFlags(int(msg._properties["trace_flags"])),
    )
    ctx = set_span_in_context(NonRecordingSpan(span_context))
    with app.tracer.start_as_current_span("consume_example", context=ctx) as span:
        logging.info(f"Consuming {msg} from {app.settings.pulsar_topic_consume}")
        await asyncio.sleep(0.3 * random())
        msg2 = ExampleSimple(msg=msg.msg, num=msg.num * 10)
        msg2._properties["trace_id"] = str(span.get_span_context().trace_id)
        msg2._properties["span_id"] = str(span.get_span_context().span_id)
        msg2._properties["trace_flags"] = str(span.get_span_context().trace_flags)
        await app.producer.send(msg2)
        logging.info(f"Sending {msg2} to {app.settings.pulsar_topic_produce}")


time.sleep(Settings().sleep)
app = Application(lifespan=lifespan)


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
