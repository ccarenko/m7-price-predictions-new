import os

from service_utils.alerts import Alerting
from service_utils.tracing import get_tracer

service = "m7-price-predictions"
tracer = get_tracer(service)
alert = Alerting(
    service=service,
    tracer=tracer,
    pagerduty_api_key=os.environ.get("PD_API_KEY"),
    slack_token=os.environ.get("SLACK_TOKEN"),
)
