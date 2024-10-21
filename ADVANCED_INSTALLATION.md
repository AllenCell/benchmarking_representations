# Installation and usage with pdm
1. [Install pdm](https://pdm-project.org/en/latest/#recommended-installation-method)
2. Install dependencies: `pdm sync --no-isolation`. (The `--no-isolation` flag is required for `torch-scatter`.)
3. Prefix every `python` command with `pdm run`. For example:
```
pdm run python br/models/train.py experiment=cellpack/pc_equiv
```
