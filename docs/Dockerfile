FROM python:3.9.15-slim AS builder

RUN \
  apt-get -y update && \
  apt-get -y upgrade && \
  apt-get --no-install-recommends -y install git ssh postgresql postgresql-client gcc python3-dev musl-dev libffi-dev python3-dev libpq-dev git

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install -U pip && pip install psycopg2==2.8.6


RUN mkdir /root/.ssh
COPY id_rsa /root/.ssh/id_rsa
RUN \
  touch /root/.ssh/known_hosts && \
  ssh-keyscan gitlab.com >> /root/.ssh/known_hosts && \
  git config --global url."git@gitlab.com:".insteadOf "https://gitlab.com/"

# Install the core deps
COPY requirements_full.txt /
RUN pip install -r /requirements_full.txt && pip install -U poetry

FROM python:3.9.15-slim
RUN \
  apt-get update && \
  apt-get install --no-install-recommends --yes postgresql \
  amazon-ecr-credential-helper \
  ssh \
  git \
  && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY docker/config.json /root/.docker/config.json
ENV PATH="/opt/venv/bin:$PATH"
COPY ./ /arenkods
RUN cd /arenkods && poetry install
