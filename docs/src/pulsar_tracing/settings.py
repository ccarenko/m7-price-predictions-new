import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    pulsar_host: str = Field(default="pulsar://pulsar:6650")
    pulsar_tenant: str = Field(default="public")
    pulsar_namespace: str = Field(default="default")

    pulsar_topic_consume: str = Field(default="A")
    pulsar_topic_produce: str = Field(default="B")

    tempo_endpoint: str = Field(default="http://tempo:4317")
    app_name: str = Field(default="example")
    sleep: int = 0
    model_config = SettingsConfigDict(case_sensitive=False)

    @property
    def full_topic_consumer(self) -> str:
        return self._get_topic(self.pulsar_topic_consume)

    @property
    def full_topic_producer(self) -> str:
        return self._get_topic(self.pulsar_topic_produce)

    def _get_topic(self, topic: str) -> str:
        return f"persistent://{self.pulsar_tenant}/{self.pulsar_namespace}/{topic}"


if __name__ == "__main__":  # pragma: no cover
    settings = Settings()
    logging.info(settings.model_dump_json(indent=4))
