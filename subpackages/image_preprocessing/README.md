# Single cell image preprocessing

Code for alignment, masking, and registration of 3D single cell images.

# Installation

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

# Usage
Once data is downloaded and config files are set up, run preprocessing scripts.
```bash
snakemake -s Snakefile --cores all
```