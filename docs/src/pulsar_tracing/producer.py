import asyncio
import logging
from random import choice, randint

from definitions.models.example import ExampleSimple
from definitions.pulsar_client import Producer, PulsarClient

from pulsar_tracing.settings import Settings
from pulsar_tracing.tracing import get_tracer

service = "example"
settings = Settings(app_name="first-producer")
msgs = [x for x in "abcdefghijklmnopqrstuv"]
tracer = get_tracer(settings.app_name, settings.tempo_endpoint)


async def main():
    pulsar_client = PulsarClient(settings.pulsar_host)
    producer = pulsar_client.create_producer(settings.full_topic_consumer, ExampleSimple, timeout_seconds=1.0)
    while True:
        await produce(producer)
        await asyncio.sleep(1.0)


async def produce(producer: Producer) -> None:
    with tracer.start_as_current_span(settings.app_name) as t:
        example = ExampleSimple(msg=choice(msgs), num=randint(1, 100))
        ctx = t.get_span_context()
        example._properties["trace_id"] = str(ctx.trace_id)
        example._properties["span_id"] = str(ctx.span_id)
        example._properties["trace_flags"] = str(ctx.trace_flags)
        # Add attributes to the span
        t.set_attribute("msg", example.msg)
        t.add_event("first send", {"event_message": example.msg})
        logging.info(f"First send of {example} into topic {settings.full_topic_consumer}")
        await asyncio.sleep(0.1)
        await producer.send(example)


if __name__ == "__main__":
    log_format = (
        "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] "
        "[trace_id=%(otelTraceID)s span_id=%(otelSpanID)s resource.service.name=%(otelServiceName)s] - %(message)s"
    )
    logging.basicConfig(level=logging.INFO, format=log_format)
    # start event loop if needed
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio.run(main())
