# Template

## Instructions

1. If your project has an image created through the `ecr-repositories` repo `main.tf` file.
    1. Modify the `Makefile` to point to that address
    2. Modify the `.gitlab-ci.yml.rename` by removing the `.rename` at the end of the file, and edit the file to point to that ECR. Uncomment the `include` section of the pipeline to get pipelines running again.
2. In `pyproject.toml`, you may want to change quite a few things, such as the author, the package, etc
3. The `poetry.toml` by default creates an in-project venv. If you want use a pre-existing venv, edit this file
4. You'll probably need to copy an `id_rsa` key into the repo, plus customise the final build steps in the `Dockerfile`

## Repo Structure (inside imbalance-price-predictions)

The below structure was generated using `tree  -L 2 -I "_*"`

```bash
.
├── Dockerfile              # Builds your image
├── Makefile                # Shortcuts to make life easy
├── README.md               # This file
├── poetry.lock             # A working snapshot of all dependencies to install
├── poetry.toml             # How poetry should be run. This file often isnt committed
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