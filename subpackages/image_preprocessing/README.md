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

# Usage
## Steps to download and preprocess data

1. Datasets are hosted on quilt. Download raw data at the following links

* [cellPACK synthetic dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/cellPACK_single_cell_punctate_structure/)
* [DNA replication foci dataset](https://open.quiltdata.com/b/allencell/packages/aics/nuclear_project_dataset_4)
* [WTC-11 hIPSc single cell image dataset v1](https://staging.allencellquilt.org/b/allencell/tree/aics/hipsc_single_cell_image_dataset/)
* [Nucleolar drug perturbation dataset](https://open.quiltdata.com/b/allencell/tree/aics/NPM1_single_cell_drug_perturbations/)

> [!NOTE]  
> Ensure to download all the data in the same folder where the repo was cloned!

2. Once data is downloaded, run preprocessing scripts to create the final image/pointcloud/SDF datasets (this step is not necessary for the cellPACK dataset). For image preprocessing used for punctate structures, install [snakemake](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html) and update the data paths in

```
└── data
    └── preprocessing
        └── image_preprocessing
            └── config
                └── config.yaml     <- Data config for image processing workflow
```

Then follow the [installation](src/br/data/preprocessing/image_preprocessing/README.md) steps to run the snakefile located in

```
└── data
    └── preprocessing
        └── image_preprocessing
            └── Snakefile      <- Image preprocessing workflow. Combines alignment, masking, registration
```

For point cloud preprocessing for punctate structures, update data paths and run the workflow in

```└
└── data
    └── preprocessing
        └── pc_preprocessing
            └── punctate_cyto.py      <- Point cloud sampling from raw images for punctate structures here
```

For SDF preprocessing for polymorphic structures, update data paths and run the workflows in

```
└── data
    └── preprocessing
        └── sdf_preprocessing
            ├── image_sdfs.py      <- Create 32**3 resolution SDF images
            └── pc_sdfs.py      <- Sample point clouds from 32**3 resolution SDF images
```

In all cases, create a single cell manifest (e.g. csv, parquet) for each dataset with a column corresponding to final processed paths, and create a split column corresponding to train/test/validation split.
