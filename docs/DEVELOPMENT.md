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