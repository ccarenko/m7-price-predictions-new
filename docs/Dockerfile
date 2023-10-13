FROM python:3.11-slim AS builder
# if a specific python version is needed change the above to FROM python:3.<minor_version>.<patch_version>-slim
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
FROM python:3.11-slim
# if a specific python version is needed change the above to FROM python:3.<minor_version>.<patch_version>-slim
ENV PYTHONUNBUFFERED=1
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create user with home directory
RUN useradd -mrs /bin/bash runner

# Copy in files
WORKDIR /home/runner
COPY src/ ./src

# Ensure Python can find our code
ENV PYTHONPATH="/home/runner/src"
WORKDIR /home/runner/src

# Recursively change the owner of the newly created /home/runner directory and give write permission
RUN chown -R runner:runner /home/runner && chmod u+w /home/runner
USER runner

# This is the part you will probably need to change the most.
# For most images that run a service you will need to run here the entrypoint script
# RUN chmod +x ./start.sh # if you need to make start.sh executable
# CMD [ "./start.sh" ] # Or whatever you need to run here
