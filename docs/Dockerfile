FROM python:3.10-slim AS builder
ENV PYTHONUNBUFFERED=1

RUN apt-get -y update && apt-get -y upgrade && apt-get --no-install-recommends -y install git ssh
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install -U pip poetry

# Set up gitlab authentication
RUN mkdir /root/.ssh
COPY id_rsa /root/.ssh/id_rsa
RUN \
  touch /root/.ssh/known_hosts && \
  ssh-keyscan gitlab.com >> /root/.ssh/known_hosts && \
  git config --global url."git@gitlab.com:".insteadOf "https://gitlab.com/"

# Install the core deps
COPY poetry.lock pyproject.toml /
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root && rm -f /root/.ssh/id_rsa

# Create final image and transfer deps
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# This is the part you will probably need to change the most
COPY . /
RUN poetry install --only-root
