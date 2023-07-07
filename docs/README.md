# Template

## Instructions

1. If your project will push an image up into the cloud for deployment, you will need to modify the `ecr-repositories` repo's `main.tf` file.
   1. [Click here to go to the file, linked to the a ds repo](https://gitlab.com/arenko-group/terraform/ecr-repositories/-/blob/develop/terraform/main.tf#L360)
   2. Create a MR for that repo with a new Elastic Container Registry for your desired image name. Ie copy and paste and change the module name and repo name
   3. Modify the `Makefile` repo (the second line of the file) to point to that repository tag
   4. Modify the `.gitlab-ci.yml.rename` by removing the `.rename` at the end of the file, and edit the file to point to that ECR. Uncomment the `include` section of the pipeline to get pipelines running again.
2. In `pyproject.toml`, you may want to change quite a few things, such as the author, the package, etc
3. The `poetry.toml` by default creates an in-project venv. (When adding poetry to an existing repo, copy this file and the `pyproject.toml`)
4. You'll probably need to copy an `id_rsa` key into the repo, plus customise the final build steps in the `Dockerfile`

## Repo Structure (inside imbalance-price-predictions)

The below structure was generated using `tree  -L 2 -I "_*"`

```bash
.
├── Dockerfile              # Builds your image
├── Makefile                # Shortcuts to make life easy
├── README.md               # This file
├── poetry.lock             # A working snapshot of all dependencies to install
├── poetry.toml             # How poetry should be run.
├── pyproject.toml          # Project details
├── src                     # All your code in here
│   └── package             # With packages nested like so
└── test                    # All your tests in here
    └── test_example.py     # Feel free to use the same folder structure in src in test
```

## Template features

* precommit with black, isort, flake, and other checks
* pytest configured and ready to run
* gitignore and dockerignore that should cover the majority of your needs
* Dockerfile that should also get you 90% of the way there
* poetry configured to work almost out of the box
* handy makefile with common commands