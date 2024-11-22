# benchmarking_representations

Code for training and benchmarking morphology appropriate representation learning methods.

# Preprocessing
This README gives instructions for running our analysis against preprocessed image data. Because the preprocessing can take a long time, we are publishing preprocessed data along with the original movies. To do the preprocessing yourself, see the instructions in [subpackages/image_preprocessing/README.md](./subpackages/image_preprocessing/README.md).

# Installation

To install and use this software, you need:
* A GPU running CUDA 11.7 (other CUDA versions may work, but they are not officially supported),
* [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (or Python 3.10 and [pdm](https://pdm-project.org/)), and
* [git](https://github.com/git-guides/install-git).

First, clone this repository.
```bash
git clone https://github.com/AllenCell/benchmarking_representations
cd benchmarking_representations
```

Create a virtual environment.

```bash
conda create --name br python=3.10
conda activate br
```

Depending on your GPU set-up, you may need to set the `CUDA_VISIBLE_DEVICES` [environment variable](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/).
To achieve this, you will first need to get the Universally Unique IDs for the GPUs and then set `CUDA_VISIBLE_DEVICES` to some/all of those (a comma-separated list), as in the following examples.

**Example 1**
```bash
export CUDA_VISIBLE_DEVICES=0,1
```

**Example 2:** Using one partition of a MIG partitioned GPU
```bash
export CUDA_VISIBLE_DEVICES=MIG-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

Next, install all required packages

```bash
pip install -r requirements1.txt
pip install -r requirements2.txt
pip install -e .
```

For `pdm` users, follow [these installation steps instead](./ADVANCED_INSTALLATION.md).

## Troubleshooting
**Q:** When installing dependencies, pytorch fails to install with the following error message.
```bash
torch.cuda.DeferredCudaCallError: CUDA call failed lazily at initialization with error: device >= 0 && device < num_gpus
```

**A:** You may need to configure the `CUDA_VISIBLE_DEVICES` [environment variable](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/).

## Set env variables
To run the models, you must set the `CYTODL_CONFIG_PATH` environment variable to point to the `br/configs` folder.
Check that your current working directory is the `benchmarking_representations` folder, then run the following command (this will last for only the duration of your shell session).
```bash
export CYTODL_CONFIG_PATH=$PWD/configs/
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
├── data
│   ├── preprocessing
│   │   ├── image_preprocessing
│   │   │   ├── config
│   │   │   │   ├── config.yaml     <- Data config for image processing workflow
```

Then follow the [installation](src/br/data/preprocessing/image_preprocessing/README.md) steps to run the snakefile located in

```
├── data
│   ├── preprocessing
│   │   ├── image_preprocessing
│   │   │   ├── Snakefile      <- Image preprocessing workflow. Combines alignment, masking, registration
```

For point cloud preprocessing for punctate structures, update data paths and run the workflow in

```
├── data
│   ├── preprocessing
│   │   ├── pc_preprocessing
│   │   │   ├── punctate_cyto.py      <- Point cloud sampling from raw images for punctate structures here
```

For SDF preprocessing for polymorphic structures, update data paths and run the workflows in

```
├── data
│   ├── preprocessing
│   │   ├── sdf_preprocessing
│   │   │   ├── image_sdfs.py      <- Create 32**3 resolution SDF images
│   │   │   ├── pc_sdfs.py      <- Sample point clouds from 32**3 resolution SDF images
```

In all cases, create a single cell manifest (e.g. csv, parquet) for each dataset with a column corresponding to final processed paths, and create a split column corresponding to train/test/validation split.

## Steps to train models

1. Update the final single cell dataset path (`SINGLE_CELL_DATASET_PATH`) and the column in the manifest for appropriate input modality (`SDF_COLUMN`/`SEG_COLUMN`/`POINTCLOUD_COLUMN`/`IMAGE_COLUMN`) in each datamodule yaml files. e.g. for PCNA data these yaml files are located here -

```
├── configs
│   ├── data
│   │   ├── pcna
│   │   │   ├── image.yaml      <- Datamodule for PCNA images
│   │   │   ├── pc.yaml       <- Datamodule for PCNA point clouds
│   │   │   ├── pc_intensity.yaml       <- Datamodule for PCNA point clouds with intensity
│   │   │   ├── pc_intensity_jitter.yaml       <- Datamodule for PCNA point clouds with intensity and jitter
```

2. Train models using cyto_dl. Ensure to run the training scripts from the folder where the repo was cloned (and where all the data was downloaded). Experiment configs for point cloud and image models are located here -

```
├── configs
│   ├── experiment
│   │   ├── cellpack
│   │   │   ├── image_equiv.yaml      <- Rotation invariant image model experiment
│   │   │   ├── pc_equiv.yaml       <- Rotation invariant point cloud model experiment
```

Here is an example of training a rotation invariant point cloud model

```bash
python src/br/models/train.py experiment=cellpack/pc_so3
```

Override parts of the experiment config via command line or manually in the configs. For example, to train a classical model, run

```bash
python src/br/models/train.py experiment=cellpack/pc_so3 model=pc/classical_earthmovers_sphere ++csv.save_dir=[SAVE_DIR]
```

## Steps to download pre-trained models and pre-computed embeddings

1. To skip model training, download pre-trained models

* [cellPACK synthetic dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_checkpoints/cellpack/)
* [DNA replication foci dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_checkpoints/pcna/)
* [WTC-11 hIPSc single cell image dataset v1 punctate structures](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_checkpoints/other_punctate/)
* [WTC-11 hIPSc single cell image dataset v1 nucleolus (NPM1)](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_checkpoints/npm1/)
* [WTC-11 hIPSc single cell image dataset v1 polymorphic structures](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_checkpoints/other_polymorphic/)
* [Nucleolar drug perturbation dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_checkpoints/npm1_perturb/)

2. Download pre-computed embeddings

* [cellPACK synthetic dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_embeddings/cellpack/)
* [DNA replication foci dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_embeddings/pcna/)
* [WTC-11 hIPSc single cell image dataset v1 punctate structures](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_embeddings/other_punctate/)
* [WTC-11 hIPSc single cell image dataset v1 nucleolus (NPM1)](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_embeddings/npm1/)
* [WTC-11 hIPSc single cell image dataset v1 polymorphic structures](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_embeddings/other_polymorphic/)
* [Nucleolar drug perturbation dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_embeddings/npm1_perturb/)

## Steps to run benchmarking analysis

1. Run analysis for each dataset separately via jupyter notebooks

```
├── br
│   ├── notebooks
│   │   ├── fig2_cellpack.ipynb      <- Reproduce Fig 2 cellPACK synthetic data results
│   │   ├── fig3_pcna.ipynb      <- Reproduce Fig 3 PCNA data results
│   │   ├── fig4_other_punctate.ipynb      <- Reproduce Fig 4 other puntate structure data results
│   │   ├── fig5_npm1.ipynb      <- Reproduce Fig 5 npm1 data results
│   │   ├── fig6_other_polymorphic.ipynb      <- Reproduce Fig 6 other polymorphic data results
│   │   ├── fig7_drug_data.ipynb      <- Reproduce Fig 7 drug data results
```

# Development
## Project Organization

```
benchmarking_representations
├── LICENSE
├── README.md      <- The top-level README for developers using this project.
├── configs        <- Training configs for each experiment
│   ├── callbacks  <- e.g. Early stopping, model checkpoint etc
│   ├── data       <- Datamodules for each dataset
│   ├── experiment <- training config for an experiment combining data, models, logger
│   ├── model      <- config for Pytorch Lightning model
│   ├── trainer    <- trainer parameters for Pytorch Lightning
│   ├── logger     <- Choice of logger to save results
│   └── hydra      <- Hydra params to perform experiment sweeps
├── src            <- Source code for data analysis and model training and inference
│   ├── br
│   │   ├── cellpack
│   │   ├── chandrasekaran_et_al
│   │   ├── data
│   │   │   ├── get_datamodules.py     <- Get final list of datamodules per dataset
│   │   │   └── preprocessing          <- Preprocessing scripts to generate point clouds and SDFs
│   │   ├── features
│   │   ├── features                   <- Metrics for benchmarking each model
│   │   │   ├── archetype.py           <- Archetype analysis functions
│   │   │   ├── classification.py      <- Test set classification accuracies using logistic regression classifiers
│   │   │   ├── outlier_compactness.py <- Intrinsic dimensionality calculation and outlier classification
│   │   │   ├── reconstruction.py      <- Functions for reconstruction viz across models
│   │   │   ├── regression.py          <- Linear regression test set r^2
│   │   │   ├── rotation_invariance.py <- Sensitivity to four 90 degree rotations in embedding space
│   │   │   └── plot.py                <- Polar plot viz across metrics
│   │   ├── models                     <- Training and inference scripts
│   │   │   ├── train.py               <- Training script using cyto_dl given an experiment config
│   │   │   ├── predict_model.py       <- Inference functions
│   │   │   ├── save_embeddings.py     <- Save embeddings using inference functions
│   │   │   ├── load_models.py         <- Load trained models based on checkpoint paths
│   │   │   └── compute_features.py    <- Compute multi-metric features for each model based on saved embeddings
│   │   ├── notebooks           <- Jupyter notebooks
│   │   └── visualization
│   └── pointcloudutils
│       ├── datamodules         <- Custom datamodules
│       │   └── cellpack.py     <- CellPACK data specific datamodule
│       └── networks            <- Custom networks
│           └── simple_inner.py <- Inner product decoder for SDF reconstruction
├── subpackages
│   └── image_preprocessing
│       └── pyproject.toml      <- Defines all dependencies for image_preprocessing
├── pyproject.toml              <- Defines all dependencies for code in src/ and makes project pip installable (pip install -e .) so br and point_cloud_utils can be imported
└── tests
```

______________________________________________________________________
