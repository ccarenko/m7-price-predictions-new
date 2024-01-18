FROM 596302374988.dkr.ecr.eu-west-2.amazonaws.com/datascience/base-ml:1.4.0 as builder
ENV PYTHONUNBUFFERED=1

RUN apt-get -y update && apt-get -y upgrade && apt-get --no-install-recommends -y install git ssh gcc g++ python3-dev musl-dev make
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
FROM python:3.11.7-slim
ENV PYTHONUNBUFFERED=1
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# mlflow seems to require git to operate
RUN apt-get -y update && \
  apt-get -y upgrade && \
  apt-get --no-install-recommends -y install git

# Create new user and set user home as working directory
RUN useradd -mrs /bin/bash runner
WORKDIR /home/runner/

# Copy in files
COPY src/ ./src
# Make sure Python can find our code
ENV PYTHONPATH="/home/runner/src"

# Make the runner user the owner of the WORKDIR and capable of writing files and creating directories
RUN chown -R runner:runner /home/runner && chmod u+w /home/runner
USER runner
CMD ["python", "src/m7_price_predictions/main.py"]
