# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-ff70592b-6c77-5bde-832d-88d1e18cad50"
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from br.features.archetype import AA_Fast
from br.features.plot import collect_outputs, plot, plot_stratified_pc
from br.features.reconstruction import stratified_latent_walk
from br.features.utils import (
    normalize_intensities_and_get_colormap,
    normalize_intensities_and_get_colormap_apply,
)
from br.models.compute_features import compute_features, get_embeddings
from br.models.load_models import get_data_and_models
from br.models.save_embeddings import (
    get_pc_loss,
    get_pc_loss_chamfer,
    save_embeddings,
    save_emissions,
)
from br.models.utils import get_all_configs_per_dataset

device = "cuda:0"

# %% [markdown]
# # Load data and models

# %%
# Set paths

os.chdir("../../benchmarking_representations/")
save_path = "./test_var_punctate_embeddings/"


# %%
# Util function
def get_data_and_models(dataset_name, batch_size, results_path, debug=False):
    data_list = get_data(dataset_name, batch_size, results_path, debug)
    all_models, run_names, model_sizes = load_model_from_path(
        dataset_name, results_path
    )  # default list of models in load_models.py
    return data_list, all_models, run_names, model_sizes


# %%
# Get datamodules, models, runs, model sizes

dataset_name = "other_punctate"
batch_size = 2
debug = False
results_path = "./configs/results/"
data_list, all_models, run_names, model_sizes = get_data_and_models(
    dataset_name, batch_size, results_path, debug
)

# %%
gg = pd.DataFrame()
gg["model"] = run_names
gg["model_size"] = model_sizes
gg.to_csv(save_path + "model_sizes.csv")

# %% [markdown]
# # compute embeddings and emissions

# %%
# Compute embeddings and reconstructions for each model

splits_list = ["train", "val", "test"]
meta_key = None
eval_scaled_img = [False] * 5
eval_scaled_img_params = [{}] * 5
loss_eval_list = None
sample_points_list = [True, True, False, False, False]
skew_scale = 100
save_embeddings(
    save_path,
    data_list,
    all_models,
    run_names,
    debug,
    splits_list,
    device,
    meta_key,
    loss_eval_list,
    sample_points_list,
    skew_scale,
    eval_scaled_img,
    eval_scaled_img_params,
)

# %%
# Save emission stats for each model

max_batches = 2
save_emissions(
    save_path,
    data_list,
    all_models,
    run_names,
    max_batches,
    debug,
    device,
    loss_eval_list,
    sample_points_list,
    skew_scale,
    eval_scaled_img,
    eval_scaled_img_params,
)

# %% [markdown]
# # Compute benchmarking features

# %%
# Compute multi-metric benchmarking features

keys = ["image", "image", "pcloud", "pcloud", "pcloud"]
max_embed_dim = 256
DATA_LIST = get_all_configs_per_dataset(results_path)
data_config_list = DATA_LIST[dataset_name]["data_paths"]

evolve_params = {
    "modality_list_evolve": keys,
    "config_list_evolve": data_config_list,
    "num_evolve_samples": 40,
    "compute_evolve_dataloaders": False,
    "eval_meshed_img": [False] * 5,
    "skew_scale": 100,
    "eval_meshed_img_model_type": [None] * 5,
    "only_embedding": False,
    "fit_pca": False,
}

loss_eval = get_pc_loss_chamfer()
loss_eval_list = [loss_eval] * 5
use_sample_points_list = [True, True, False, False, False]

classification_params = {"class_labels": ["structure_name", "cell_stage"]}
rot_inv_params = {"squeeze_2d": False, "id": "cell_id", "max_batches": 40}

regression_params = {"df_feat": None, "target_cols": None, "feature_df_path": None}

compactness_params = {
    "method": "mle",
    "num_PCs": None,
    "blobby_outlier_max_cc": None,
    "check_duplicates": True,
}

splits_list = ["train", "val", "test"]
compute_embeds = False

metric_list = [
    "Rotation Invariance Error",
    "Evolution Energy",
    "Reconstruction",
    "Classification",
    "Compactness",
]


compute_features(
    dataset=dataset_name,
    results_path=results_path,
    embeddings_path=save_path,
    save_folder=save_path,
    data_list=data_list,
    all_models=all_models,
    run_names=run_names,
    use_sample_points_list=use_sample_points_list,
    keys=keys,
    device=device,
    max_embed_dim=max_embed_dim,
    splits_list=splits_list,
    compute_embeds=compute_embeds,
    classification_params=classification_params,
    regression_params=regression_params,
    metric_list=metric_list,
    loss_eval_list=loss_eval_list,
    evolve_params=evolve_params,
    rot_inv_params=rot_inv_params,
    compactness_params=compactness_params,
)

# %% [markdown]
# # Polar plot viz

# %%
# Holistic viz of features
model_order = [
    "Classical_image",
    "Rotation_invariant_image",
    "Classical_pointcloud",
    "Rotation_invariant_pointcloud",
]
metric_list = [
    "reconstruction",
    "emissions",
    "classification_cell_stage",
    "classification_structure_name",
    "compactness",
    "evolution_energy",
    "model_sizes",
    "rotation_invariance_error",
]
norm = "std"
title = "variance_comparison"
colors_list = None
unique_expressivity_metrics = ["classification_cell_stage", "classification_structure_name"]
df, df_non_agg = collect_outputs(save_path, norm, model_order, metric_list)
plot(save_path, df, model_order, title, colors_list, norm, unique_expressivity_metrics)

# %% [markdown]
# # latent walks

# %%
# Load model and embeddings
run_names = ["Rotation_invariant_pointcloud_structurenorm"]
DATASET_INFO = get_all_configs_per_dataset(results_path)
all_ret, df = get_embeddings(run_names, dataset_name, DATASET_INFO, save_path)

# %%
all_ret = all_ret.merge(df[["CellId", "structure_name", "cell_stage"]], on="CellId")

# %%
structs = ["NUP153", "SON", "HIST1H2BJ", "SMC1A", "CETN2", "SLC25A17", "RAB5A"]
all_ret = all_ret.loc[all_ret["structure_name"].isin(structs)].reset_index(drop=True)

# %%
# Params for viz
key = "pcloud"
stratify_key = "structure_name"
z_max = None
z_ind = 2
flip = False
views = ["xy"]
xlim = [-20, 20]
ylim = [-20, 20]

# %%
# Compute stratified latent walk

this_save_path = Path(save_path) / Path("latent_walks")
this_save_path.mkdir(parents=True, exist_ok=True)

stratified_latent_walk(
    model,
    device,
    all_ret,
    "pcloud",
    256,
    256,
    2,
    this_save_path,
    stratify_key,
    latent_walk_range=[-2, 0, 2],
    z_max=z_max,
    z_ind=z_ind,
)

# %%
# Save reconstruction plots

viz_norms = {
    "CETN2": [440, 800],
    "NUP153": [420, 600],
    "SON": [420, 1500],
    "SMC1A": [450, 630],
    "RAB5A": [420, 600],
    "SLC25A17": [400, 515],
    "HIST1H2BJ": [450, 2885],
}
import yaml

# norms used for model training
model_norms = "./src/br/data/preprocessing/pc_preprocessing/model_structnorms.yaml"
with open(model_norms) as stream:
    model_norms = yaml.safe_load(stream)

# norms used for viz
viz_norms = "./src/br/data/preprocessing/pc_preprocessing/viz_structnorms.yaml"
with open(viz_norms) as stream:
    viz_norms = yaml.safe_load(stream)

import os

items = os.listdir(this_save_path)
for struct in structs:
    fnames = [i for i in items if i.split(".")[-1] == "csv"]
    fnames = [i for i in fnames if i.split("_")[1] == "0"]
    fnames = [i for i in fnames if i.split("_")[0] in [struct]]
    names = [i.split(".")[0] for i in fnames]

    renorm = model_norms[struct]
    this_viz_norm = viz_norms[struct]
    use_vmin = this_viz_norm[0]
    use_vmax = this_viz_norm[1]

    all_df = []
    for idx, _ in enumerate(fnames):
        fname = fnames[idx]
        df = pd.read_csv(f"{this_save_path}/{fname}", index_col=0)
        df["s"] = df["s"] / 10  # scalar values were scaled by 10 during training
        df["s"] = df["s"] * (renorm[1] - renorm[0]) + renorm[0]
        df[stratify_key] = names[idx]
        all_df.append(df)
    df = pd.concat(all_df, axis=0).reset_index(drop=True)
    if struct in ["NUP153", "SON", "HIST1H2BJ", "SMC1A"]:
        df = df.loc[df["z"] < 0.2].reset_index(drop=True)
    df = normalize_intensities_and_get_colormap_apply(df, use_vmin, use_vmax)
    plot_stratified_pc(df, xlim, ylim, stratify_key, this_save_path, cmap, flip)

    for pc_bin in df["structure_name"].unique():
        this_df = df.loc[df["structure_name"] == pc_bin].reset_index(drop=True)
        print(this_df.shape, struct, pc_bin)
        np_arr = this_df[["x", "y", "z"]].values
        colors = cmap(this_df["inorm"].values)[:, :3]
        np_arr2 = colors
        np_arr = np.concatenate([np_arr, np_arr2], axis=1)
        np.save(this_save_path / Path(f"{stratify_key}_{pc_bin}.npy"), np_arr)
        cmap = plt.get_cmap("YlGnBu")

# %%
