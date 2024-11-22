# Single cell image preprocessing

Code for preprocessing 3D single cell images

## Installation

Move to this `image_preprocessing` directory.
```bash
cd subpackages/image_preprocessing
```

Install dependencies.
```bash
conda create --name preprocessing_env python=3.10
conda activate preprocessing_env
pip install -r requirements.txt
pip install -e .
```

## Run workflow

```bash
snakemake -s Snakefile --cores ..
```
