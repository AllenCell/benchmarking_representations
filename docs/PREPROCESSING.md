# Preprocessing

Preprocessing is divided into three steps that use two different virtual environments.

1. Punctate structures: Alignment, masking, and registration (`image_preprocessing` virtual environment)
2. Punctate structures: Generate pointclouds (main virtual environment)
3. Polymorphic structures: Generate SDFs (main virtual environment)

# System requirements

- A GPU running CUDA 11.7 (other CUDA versions may work, but they are not officially supported),
- OpenGL,
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (or Python 3.10 and [pdm](https://pdm-project.org/)), and
- [git](https://github.com/git-guides/install-git).

# Configure input data

1. Datasets are hosted on quilt. Download raw data at the following links

- [cellPACK synthetic dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/cellPACK_single_cell_punctate_structure/)
- [DNA replication foci dataset](https://open.quiltdata.com/b/allencell/packages/aics/nuclear_project_dataset_4)
- [WTC-11 hIPSc single cell image dataset v1](https://staging.allencellquilt.org/b/allencell/tree/aics/hipsc_single_cell_image_dataset/)
- [Nucleolar drug perturbation dataset](https://open.quiltdata.com/b/allencell/tree/aics/NPM1_single_cell_drug_perturbations/)

> [!NOTE]
> Ensure to download all the data in the `benchmarking_representations` folder.

# Punctate structures: Alignment, masking, and registration

1. Edit the data paths in the file `subpackages/image_preprocessing/config/config.yaml` to point to your copies of the data.
2. Follow the [installation and usage instructions](/subpackages/image_preprocessing/README.md) to create the `image_preprocessing` virtual environment and run the Snakefile.

# Switch to main virtual environment

1. Deactivate the `image_preprocessing` virtual environment (if applicable).
2. Follow the [installation instructions](./USAGE.md) (everything before "Usage") for the main virtual environment.

# Punctate structures: Generate pointclouds

Edit the data paths in the following file to match the location of the outputs of the alignment, masking, and registration step, then run it.

```
src
└── br
    └── data
        └── preprocessing
            └── pc_preprocessing
                └── punctate_cyto.py <- Point cloud sampling from raw images for punctate structures here
```

# Polymorphic structures: Generate SDFs

Edit the data paths in the following files to match the location of your copy of the data, then run both.

```
src
└── br
    └── data
       └── preprocessing
           └── sdf_preprocessing
               ├── image_sdfs.py <- Create 32**3 resolution SDF images
               └── pc_sdfs.py    <- Sample point clouds from 32**3 resolution SDF images
```
