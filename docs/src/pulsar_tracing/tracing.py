from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, Tracer


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


def get_tracer(app_name: str, endpoint: str, log_correlation: bool = True) -> Tracer:
    resource = Resource.create(attributes={"service.name": app_name})

    # set the tracer provider
    tracer = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer)

    tracer.add_span_processor(MySpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

    if log_correlation:
        LoggingInstrumentor().instrument(set_logging_format=True)

    return trace.get_tracer(app_name)
