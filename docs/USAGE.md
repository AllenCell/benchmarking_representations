# Installation

To install and use this software, you need:

- A GPU running CUDA 11.7 (other CUDA versions may work, but they are not officially supported),
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (or Python 3.10 and [pdm](https://pdm-project.org/)), and
- [git](https://github.com/git-guides/install-git).

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

# 1. Model training

## Steps to download pre-processed data

Coming soon.

## Steps to train models

Training these models can take days. We've published our trained models so you don't have to. Skip to the [next section](#2-model-inference) if you'd like to just use our models.

1. Create a single cell manifest (e.g. csv, parquet) for each dataset with a column corresponding to final processed paths, and create a split column corresponding to train/test/validation split.
2. Update the final single cell dataset path (`SINGLE_CELL_DATASET_PATH`) and the column in the manifest for appropriate input modality (`SDF_COLUMN`/`SEG_COLUMN`/`POINTCLOUD_COLUMN`/`IMAGE_COLUMN`) in each [datamodule file](../configs/data/). e.g. for PCNA data these yaml files are located here -

```
configs
└── data
    └── pcna
        ├── image.yaml               <- Datamodule for PCNA images
        ├── pc.yaml                  <- Datamodule for PCNA point clouds
        ├── pc_intensity.yaml        <- Datamodule for PCNA point clouds with intensity
        └── pc_intensity_jitter.yaml <- Datamodule for PCNA point clouds with intensity and jitter
```

3. Train models using cyto_dl. Ensure to run the training scripts from the folder where the repo was cloned (and where all the data was downloaded). [Experiment configs](../configs/experiment/) for point cloud and image models for the cellpack dataset are located here:

```
└── configs
    └── experiment
        └── cellpack
            ├── image_classical.yaml <- Classical image model experiment
            ├── image_so3.yaml <- Rotation invariant image model experiment
            └── pc_classical.yaml    <- Classical point cloud model experiment
            └── pc_so3.yaml    <- Rotation invariant point cloud model experiment
```

Here is an example of training a rotation invariant point cloud model.

```bash
python src/br/models/train.py experiment=cellpack/pc_so3
```

Override parts of the experiment config via command line or manually in the configs. For example, to train a classical model, run the following.

```bash
python src/br/models/train.py experiment=cellpack/pc_so3 model=pc/classical_earthmovers_sphere ++csv.save_dir=[SAVE_DIR]
```

# 2. Model inference

## Steps to download pre-trained models

1. To skip model training, download pre-trained models. For each of the six datasets, there are five `.ckpt` files to download.

| Download `.ckpt` from here                                                                                                                                                                               | To this folder                               |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| [cellPACK synthetic dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_checkpoints/cellpack/)                                                | `configs/experiment/cellpack/ckpts`          |
| [DNA replication foci dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_checkpoints/pcna/)                                                  | `configs/experiment/pcna/ckpts`              |
| [WTC-11 hIPSc single cell image dataset v1 punctate structures](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_checkpoints/other_punctate/)       | `configs/experiment/other_punctate/ckpts`    |
| [WTC-11 hIPSc single cell image dataset v1 nucleolus (NPM1)](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_checkpoints/npm1/)                    | `configs/experiment/npm1/ckpts`              |
| [WTC-11 hIPSc single cell image dataset v1 polymorphic structures](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_checkpoints/other_polymorphic/) | `configs/experiment/other_polymorphic/ckpts` |
| [Nucleolar drug perturbation dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_checkpoints/npm1_perturb/)                                   | `configs/experiment/npm1_perturb/ckpts`      |

## Compute embeddings
To compute embeddings from the trained models, update the data paths in the [datamodule files](../configs/data/) to point to your pre-processed data.
Then, run the following commands.

Dataset | Embedding command
--------|---
cellpack | `python src/br/analysis/run_embeddings.py --save_path "./outputs/" --sdf False --dataset_name cellpack --batch_size 5 --debug False`
npm1_perturb | `python src/br/analysis/run_embeddings.py --save_path "./outputs/" --sdf True --dataset_name npm1_perturb --batch_size 5 --debug False`
npm1 | `python src/br/analysis/run_embeddings.py --save_path "./outputs/" --sdf True --dataset_name npm1 --batch_size 5 --debug False`
other_polymorphic | `python src/br/analysis/run_embeddings.py --save_path "./outputs/" --sdf True --dataset_name other_polymorphic --batch_size 5 --debug False`
other_punctate | `python src/br/analysis/run_embeddings.py --save_path "./outputs/" --sdf False --dataset_name other_punctate --batch_size 5 --debug False`
pcna | `python src/br/analysis/run_embeddings.py --save_path "./outputs/" --sdf False --dataset_name pcna --batch_size 5 --debug False`

# 3. Interpretability analysis
## Steps to download pre-computed embeddings

Many of the results from the paper can be reproduced just from the embeddings produced by the model. However, some results rely on statistics about the costs of running the models, which are not included with the embeddings.

You can download our pre-computed embeddings here.

- [cellPACK synthetic dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_embeddings/cellpack/)
- [DNA replication foci dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_embeddings/pcna/)
- [WTC-11 hIPSc single cell image dataset v1 punctate structures](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_embeddings/other_punctate/)
- [WTC-11 hIPSc single cell image dataset v1 nucleolus (NPM1)](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_embeddings/npm1/)
- [WTC-11 hIPSc single cell image dataset v1 polymorphic structures](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_embeddings/other_polymorphic/)
- [Nucleolar drug perturbation dataset](https://open.quiltdata.com/b/allencell/tree/aics/morphology_appropriate_representation_learning/model_embeddings/npm1_perturb/)

## Steps to run benchmarking analysis

1. To compute benchmarking features from the embeddings and trained models, run

```bash
python src/br/analysis/run_features.py --save_path "/outputs_cellpack/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/cellpack" --sdf False --dataset_name "cellpack" --debug False
```

where dataset_name corresponds to a [result config](../configs/results/).

2. To run analysis like latent walks and archetype analysis on the embeddings and trained models, run

```
python src/br/analysis/run_analysis.py --save_path "./outputs_cellpack/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/cellpack" --dataset_name "cellpack" --run_name "Rotation_invariant_pointcloud_jitter" --sdf False
```

3. To run drug perturbation analysis, run

```
python src/br/analysis/run_drugdata_analysis.py --save_path "./outputs_npm1_perturb/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/npm1_perturb/" --dataset_name "npm1_perturb"
```
