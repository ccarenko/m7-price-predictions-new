# Template

## Instructions

1. If your project has an image created through the `ecr-repositories` repo `main.tf` file.
    1. Modify the `Makefile` to point to that address
    2. Modify the `.gitlab-ci.yml` to point to that address
2. In `pyproject.toml`, you may want to change quite a few things, such as the author, the package, etc
3. The `poetry.toml` by default creates an in-project venv. If you want use a pre-existing venv, edit this file

## Repo Structure (inside imbalance-price-predictions)

The below structure was generated using `tree  -L 2 -I "_*" -I build -I test -I cache`

```bash
.
├── Dockerfile                 
├── README.md                  
├── src
│   ├── package               
│   └── main.py                 
├── tests             
└── requirements.txt           
```
