import time
from functools import wraps
from typing import TypeVar

from definitions.models.base import SchemaBase
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import NonRecordingSpan, SpanContext, SpanKind, TraceFlags, Tracer, set_span_in_context

from pulsar_tracing.prom import EXCEPTIONS, REQUESTS, REQUESTS_IN_PROGRESS, REQUESTS_PROCESSING_TIME, RESPONSES
from pulsar_tracing.settings import Settings


class MySpanProcessor(BatchSpanProcessor):
    def on_end(self, span: ReadableSpan) -> None:
        if span.kind == SpanKind.INTERNAL and span.attributes is not None:
            span_type = span.attributes.get("type", None)
            if span_type in (
                "http.request",
                "http.response.start",
                "http.response.body",
            ):
                return
        super().on_end(span=span)


T = TypeVar("T", bound=SchemaBase)


def get_tracer(app_name: str, endpoint: str, log_correlation: bool = True) -> Tracer:
    resource = Resource.create(attributes={"service.name": app_name})

    # set the tracer provider
    tracer = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer)

    tracer.add_span_processor(MySpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

    if log_correlation:
        LoggingInstrumentor().instrument(set_logging_format=True)

    return trace.get_tracer(app_name)


_s = Settings()
tracer = get_tracer(_s.app_name, _s.tempo_endpoint)


def trace_reader(fn):
    name = fn.__name__

    @wraps(fn)
    async def wrapper(model: SchemaBase, *args, **kwargs) -> None:
        ctx = None
        if {"trace_id", "span_id", "trace_flags"}.issubset(model._properties):
            span_context = SpanContext(
                trace_id=int(model._properties["trace_id"]),
                span_id=int(model._properties["span_id"]),
                is_remote=True,
                trace_flags=TraceFlags(int(model._properties["trace_flags"])),
            )
            ctx = set_span_in_context(NonRecordingSpan(span_context))
        with tracer.start_as_current_span(name, context=ctx):
            REQUESTS_IN_PROGRESS.labels(name=name, app_name=_s.app_name).inc()
            REQUESTS.labels(name=name, app_name=_s.app_name).inc()
            before_time = time.perf_counter()
            try:
                res = await fn(model, *args, **kwargs)
            except BaseException as e:
                EXCEPTIONS.labels(
                    name=name,
                    exception_type=type(e).__name__,
                    app_name=_s.app_name,
                ).inc()
                raise
            else:
                after_time = time.perf_counter()
                REQUESTS_PROCESSING_TIME.labels(name=name, app_name=_s.app_name).observe(after_time - before_time)
            finally:
                RESPONSES.labels(name=name, app_name=_s.app_name).inc()
                REQUESTS_IN_PROGRESS.labels(name=name, app_name=_s.app_name).dec()
            return res

    return wrapper


def add_tracing(x: T) -> T:
    ctx = trace.get_current_span().get_span_context()
    if ctx is None:
        return x
    x._properties["trace_id"] = str(ctx.trace_id)
    x._properties["span_id"] = str(ctx.span_id)
    x._properties["trace_flags"] = str(ctx.trace_flags)
    return x
