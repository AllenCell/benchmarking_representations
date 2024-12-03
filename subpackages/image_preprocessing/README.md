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

# Configuration

Edit the data paths in the file `subpackages/image_preprocessing/config/config.yaml` to point to your copies of the data.

# Usage

Once data is downloaded and config files are set up, run preprocessing scripts. To run the preprocessing for punctate structures from the WTC-11 hIPS single cell image dataset v1, run the following command

```bash
snakemake --cores all
```

To run the preprocessing for the DNA replication foci dataset, run the following command

```bash
snakemake -s Snakefile_pcna --cores all
```

# Troubleshooting

Before running the script, please ensure the following:

1. **Set the `TMPDIR`:**
   You need to set the `TMPDIR` environment variable, as the Snakefile requires a temporary directory. You can do this by executing:

   ```bash
   export TMPDIR=/path/to/your/tmpdir
   ```

2. **Set the `output_dir`:**
   Additionally, you must specify the `output_dir` required by [this](https://github.com/AllenCell/benchmarking_representations/blob/pcna_preprocessing/subpackages/image_preprocessing/config_pcna/config.yaml) config file.
