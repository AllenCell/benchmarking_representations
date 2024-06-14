# benchmarking_representations

Code for training and evaluating morphology appropriate representation learning methods.

## Installation

```bash
pip install numpy
pip install cyto-dl[all] git+https://github.com/AllenCellModeling/cyto-dl@br_release
pip install -e .[all]
pip install -e ./pointcloudutils
```

## Set env variables

```bash
[optional] export CUDA_VISIBLE_DEVICES=...
export CYTODL_CONFIG_PATH=./br/configs/
```

## Project Organization

```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.

├── br                <- Source code for use in this project.
│   ├── data
│   │   ├── preprocessing        <- Preprocessing scripts to generate point clouds and SDFs
│   │   ├── get_datamodules.py      <- Get final list of datamodules per dataset
│
│   ├── models             <- Training and inference scripts
│   │   ├── train.py       <- Training script using cyto_dl given an experiment config
│   │   ├── predict_model.py        <- Inference functions
│   │   ├── save_embeddings.py      <- Save embeddings using inference functions
│   │   ├── load_models.py      <- Load trained models based on checkpoint paths
│   │   ├── compute_features.py      <- Compute multi-metric features for each model based on saved embeddings
│
│   ├── features             <- Metrics for benchmarking each model
│   │   ├── archetype.py       <- Archetype analysis functions
│   │   ├── classification.py        <- Test set classification accuracies using logistic regression classifiers
│   │   ├── outlier_compactness.py      <- Intrinsic dimensionality calculation and outlier classification
│   │   ├── reconstruction.py      <- Functions for reconstruction viz across models
│   │   ├── regression.py      <- Linear regression test set r^2
│   │   ├── rotation_invariance.py      <- Sensitivity to four 90 degree rotations in embedding space
│   │   ├── plot.py      <- Polar plot viz across metrics
│
├── configs          <- Training configs for each experiment
│   ├── callbacks       <- e.g. Early stopping, model checkpoint etc
│   ├── data        <- Datamodules for each dataset
│   ├── experiment      <- training config for an experiment combining data, models, logger
│   ├── model            <- config for Pytorch Lightning model
│   ├── trainer       <- trainer parameters for Pytorch Lightning
│   ├── logger        <- Choice of logger to save results
│   ├── hydra      <- Hydra params to perform experiment sweeps
│
├── notebooks          <- Jupyter notebooks.
│
├── pointcloudutils
│   ├── pointcloudutils
│   │   ├── datamodules        <- Custom datamodules
│   │   │   ├── cellpack.py      <- CellPACK data specific datamodule
│   │   ├── networks        <- Custom networks
│   │   │   ├── simple_inner.py      <- Inner product decoder for SDF reconstruction
│
├── pyproject.toml           <- makes project pip installable (pip install -e .) so br can be imported
```

______________________________________________________________________

## Steps to download data, train models, run benchmarking analysis

To download data and train models, run steps 1, 2 and 3. To skip this and run benchmarking analysis on pre-computed embeddings, skip to step 4.

1. \[Optional\] Datasets are hosted on quilt. Download raw data at the following links

```bash
[cellPACK synthetic dataset]
[DNA replication foci dataset] https://open.quiltdata.com/b/allencell/packages/aics/nuclear_project_dataset_4
[WTC-11 hIPSc single cell image dataset v1] https://staging.allencellquilt.org/b/allencell/tree/aics/hipsc_single_cell_image_dataset/
[Nucleolar drug perturbation dataset]
```

2. \[Optional\] Once data is downloaded, run preprocessing scripts to create the final image and point cloud datasets (for cellPACK synthetic dataset, we provide final versions of both). For image preprocessing used for punctate structures, install [snakemake](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html), then update the data paths in

```
├── data
│   ├── preprocessing
│   │   ├── image_preprocessing
│   │   │   ├── config
│   │   │   │   ├── config.yaml     <- Config for image processing workflow [Update data paths]
```

Then follow the [installation](br/data/preprocessing/image_preprocessing/README.md) steps to run the snakefile located in

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
│   │   │   ├── punctate_cyto.py      <- Point cloud sampling from raw images for punctate structures [Update data paths] here
```

For SDF preprocessing for polymorphic structures, update data paths and run the workflows in

```
├── data
│   ├── preprocessing
│   │   ├── sdf_preprocessing
│   │   │   ├── image_sdfs.py      <- Create 32**3 resolution SDF images [Update data paths]
│   │   │   ├── pc_sdfs.py      <- Sample point clouds from 32**3 resolution SDF images [Update data paths]
```

In all cases, create a single cell manifest for each dataset with a column corresponding to final processed paths, and create a split column corresponding to train/test/validation.

3. Update the processed data path column in the datamodule yaml files. e.g. for PCNA data these yaml files are located here -

```
├── configs
│   ├── data
│   │   ├── pcna
│   │   │   ├── image.yaml      <- Datamodule for PCNA images [Update data paths]
│   │   │   ├── pc.yaml       <- Datamodule for PCNA point clouds [Update data paths]
│   │   │   ├── pc_intensity.yaml       <- Datamodule for PCNA point clouds with intensity [Update data paths]
│   │   │   ├── pc_intensity_jitter.yaml       <- Datamodule for PCNA point clouds with intensity and jitter [Update data paths]
```

2. \[Optional\] Train models using cyto_dl. Experiment configs for point cloud and image models are located here -

```
├── configs
│   ├── experiment
│   │   ├── cellpack
│   │   │   ├── image_equiv.yaml      <- SO3 image model experiment
│   │   │   ├── pc_equiv.yaml       <- SO3 point cloud model experiment
```

Here is an example of training an SO3 point cloud model

```bash
python br/models/train.py experiment=cellpack/pc_equiv ++mlflow.experiment_name=[EXPERIMENT_NAME] ++mlflow.run_name=[RUN_NAME]
```

Override parts of the experiment config via command line or manually in the configs. For example, to train a classical model, run

```bash
python br/models/train.py experiment=cellpack/pc_equiv model=pc/classical_earthmovers_sphere ++mlflow.experiment_name=[EXPERIMENT_NAME] ++mlflow.run_name=[RUN_NAME]
```

3. \[Optional\] Alternatively, download pre-computed embeddings.

4. Run benchmarking notebooks

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
