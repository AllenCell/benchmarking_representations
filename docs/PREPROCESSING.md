# Preprocessing

Preprocessing is divided into three steps that use two different virtual environments.

1. Punctate structures: Alignment, masking, and registration (`image_preprocessing` virtual environment)
2. Punctate structures: Generate pointclouds (main virtual environment)
3. Polymorphic structures: Generate SDFs (main virtual environment)

# System requirements

- A GPU running CUDA (tested against 11.7),
- OpenGL (tested against core profile version 4.5 (Core Profile) Mesa 21.0.3),
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (or Python 3.10 and [pdm](https://pdm-project.org/)), and
- [git](https://github.com/git-guides/install-git).

# Configure input data

1. Datasets are hosted on quilt. Download raw data at the following links

- [cellPACK synthetic dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/cellPACK_single_cell_punctate_structure/)
- [DNA replication foci dataset](https://open.quiltdata.com/b/allencell/packages/aics/nuclear_project_dataset_4)
- [WTC-11 hIPSc single cell image dataset v1](https://open.quiltdata.com/b/allencell/tree/aics/hipsc_single_cell_image_dataset/)
- [Nucleolar drug perturbation dataset](https://open.quiltdata.com/b/allencell/tree/aics/NPM1_single_cell_drug_perturbations/)

> [!NOTE]
> Ensure to download all the data in the `benchmarking_representations` folder.

# Punctate structures: Alignment, masking, and registration

1. Edit the data paths in the file `subpackages/image_preprocessing/config/config.yaml` to point to your copies of the data.
2. Follow the [installation and usage instructions](/subpackages/image_preprocessing/README.md) to create the `image_preprocessing` virtual environment and run the Snakefile.

# Switch to main virtual environment

1. Deactivate the `image_preprocessing` virtual environment (if applicable).
2. Follow the [installation instructions](./USAGE.md) (everything before "1. Model training") for the main virtual environment.

# Punctate structures: Generate pointclouds

Use the preprocessed data manifest generated via the alignment, masking, and registration steps from image as input to the pointcloud generation step

```
src
└── br
    └── data
        └── preprocessing
            └── pc_preprocessing
                └── pcna.py <- Point cloud sampling from raw images for DNA replication foci dataset here
                └── punctate_nuc.py <- Point cloud sampling from raw images of nuclear structures from the WTC-11 hIPS single cell image dataset here
                └── punctate_cyto.py <- Point cloud sampling from raw images of cytoplasmic structures from the WTC-11 hIPS single cell image dataset here
```

# Polymorphic structures: Generate SDFs

Use the segmentation data for polymorphic structures as input to the SDF generation step.

```
src
└── br
    └── data
       └── preprocessing
           └── sdf_preprocessing
               ├── image_sdfs.py <- Create scaled meshes, and 32**3 resolution SDF and seg images
               ├── get_max_bounding_box.py <- Get bounds of the largest scaled mesh
               └── pc_sdfs.py    <- Sample point clouds from scaled meshes
```

The scale factors can be computed using the `get_max_bounding_box` script. Alternatively, the pre-computed scale factors can be downloaded along with the rest of the preprocessed data. The following scale factors are available for download

1. [WTC-11 hIPSc single cell image dataset v1 nucleolus (NPM1)](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/preprocessed_data/npm1/scale_factor.npz)
2. [WTC-11 hIPSc single cell image dataset v1 nucleolus (NPM1) 64 resolution](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/preprocessed_data/npm1_64_res/scale_factor.npz)
3. [WTC-11 hIPSc single cell image dataset v1 polymorphic structures](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/preprocessed_data/other_polymorphic/scale_factor.npz)
