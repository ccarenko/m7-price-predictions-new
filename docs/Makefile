.PHONY: test
REPO := datascience/YOUR_REPO_HERE

install:
	pip install -U poetry
	poetry install --with=dev,test --all-extras
	poetry run pre-commit install

update_deps:
	poetry check
	poetry lock
	poetry export --without-hashes -f requirements.txt -o requirements.txt
	poetry export --with dev,test --without-hashes -f requirements.txt -o requirements_dev.txt

build_image:
	aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin 596302374988.dkr.ecr.eu-west-2.amazonaws.com
	docker build -t $(REPO) .
	docker tag $(REPO):latest 596302374988.dkr.ecr.eu-west-2.amazonaws.com/$(REPO):latest
	docker push 596302374988.dkr.ecr.eu-west-2.amazonaws.com/$(REPO):latest

build_image_local:
	docker build -t $(REPO) .

precommit:
	poetry run pre-commit run --all-files

test:
	poetry run pytest

tests: test

all: precommit update_deps tests build_image_local
