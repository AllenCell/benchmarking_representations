# benchmarking_representations

benchmarking different methods for extracting unsupervised representations from images

## Installation

```bash
pip install numpy
pip install cyto-dl[all] git+https://github.com/AllenCellModeling/cyto-dl@br_release
pip install -e .
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
│   ├── configs          <- Training configs for each experiment
│   │   ├── callbacks       <- e.g. Early stopping, model checkpoint etc
│   │   ├── data        <- Datamodules for each dataset
│   │   ├── experiment      <- training config for an experiment combining data, models, logger etc
│   │   └── model            <- config for Pytorch Lightning model
│   │   ├── trainer       <- trainer parameters for Pytorch Lightning
│   │   ├── logger        <- Choice of logger to save results
│   │   ├── hydra      <- Hydra params to perform experiment sweeps
│
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

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
