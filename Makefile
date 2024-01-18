.PHONY: test
REPO := datascience/m7-price-predictions

install:
	pip install -U pip poetry -q
	poetry install --with=dev,test --all-extras
	poetry run pre-commit install
	poetry run pre-commit autoupdate

update_deps:
	poetry check
	poetry lock

build_image:
	docker build -t $(REPO) .

push_image:
	aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin 596302374988.dkr.ecr.eu-west-2.amazonaws.com
	docker tag $(REPO):latest 596302374988.dkr.ecr.eu-west-2.amazonaws.com/$(REPO):latest
	docker push 596302374988.dkr.ecr.eu-west-2.amazonaws.com/$(REPO):latest

precommit-update:
	poetry run pre-commit autoupdate

precommit:
	poetry run pre-commit run --all-files

test:
	poetry run pytest

tests: test

all: precommit tests build_image

image: build_image push_image

docker: build_image docker_run