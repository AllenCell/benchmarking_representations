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
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-25a8cdbf-56c0-521b-b855-e8cd1f848fa1"
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
save_path = "./test_pcna_embeddings/"

# %%
# Get datamodules, models, runs, model sizes

dataset_name = "pcna"
batch_size = 2
debug = False
results_path = "./configs/results/"
data_list, all_models, run_names, model_sizes = get_data_and_models(
    dataset_name, batch_size, results_path, debug
)

gg = pd.DataFrame()
gg["model"] = run_names
gg["model_size"] = model_sizes
gg.to_csv(save_path + "model_sizes.csv")

# %% [markdown]
# # Compute embeddings and emissions

# %%
# Compute embeddings and reconstructions for each model

splits_list = ["train", "val", "test"]
meta_key = None
eval_scaled_img = [False] * 5
eval_scaled_img_params = [{}] * 5
loss_eval_list = None
sample_points_list = [False, False, True, True, False]
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

keys = ["pcloud", "pcloud", "image", "image", "pcloud"]
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
use_sample_points_list = [False, False, True, True, False]

classification_params = {"class_labels": ["cell_stage_fine", "flag_comment"]}
rot_inv_params = {"squeeze_2d": False, "id": "cell_id", "max_batches": 4000}

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
    # "Rotation Invariance Error",
    # "Evolution Energy",
    # "Reconstruction",
    "Classification",
    # "Compactness",
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
    "classification_cell_stage_fine",
    "classification_flag_comment",
    "compactness",
    "evolution_energy",
    "model_sizes",
    "rotation_invariance_error",
]
norm = "std"
title = "pcna_comparison"
colors_list = None
unique_expressivity_metrics = ["Classification_cell_stage_fine", "Classification_flag_comment"]
df, df_non_agg = collect_outputs(save_path, norm, model_order, metric_list)
plot(save_path, df, model_order, title, colors_list, norm, unique_expressivity_metrics)

# %%
# %matplotlib inline
import seaborn as sns

sns.set(font_scale=5)
sns.set_style("white")
g = sns.catplot(
    data=df_non_agg,
    x="model",
    y="value",
    col="variable",
    kind="bar",
    sharey=False,
    sharex=True,
    order=model_order,
    col_wrap=5,
    height=20,
    aspect=1,
)
g.set_xticklabels(rotation=30)

# %% [markdown]
# # Latent walks

# %%
# Load model and embeddings
run_names = ["Rotation_invariant_pointcloud_jitter"]
DATASET_INFO = get_all_configs_per_dataset(results_path)
all_ret, df = get_embeddings(run_names, dataset_name, DATASET_INFO, save_path)
model = all_models[-1]
# Subset to interphase stages
interphase_stages = [
    "G1",
    "earlyS",
    "earlyS-midS",
    "midS",
    "midS-lateS",
    "lateS",
    "lateS-G2",
    "G2",
]
all_ret = all_ret.loc[all_ret["cell_stage_fine"].isin(interphase_stages)].reset_index(drop=True)

# %%
# Params for viz
key = "pcloud"
stratify_key = "cell_stage_fine"
z_max = 0.3
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

import os

items = os.listdir(this_save_path)
fnames = [i for i in items if i.split(".")[-1] == "csv"]
fnames = [i for i in fnames if i.split("_")[1] == "0"]
fnames = [i for i in fnames if i.split("_")[0] in interphase_stages]
names = [i.split(".")[0] for i in fnames]

all_df = []
for idx, _ in enumerate(fnames):
    fname = fnames[idx]
    df = pd.read_csv(f"{this_save_path}/{fname}", index_col=0)
    # normalize per PC
    df, cmap, vmin, vmax = normalize_intensities_and_get_colormap(
        df, pcts=[5, 95], cm_name="YlGnBu"
    )
    df[stratify_key] = names[idx]
    all_df.append(df)
df = pd.concat(all_df, axis=0).reset_index(drop=True)
plot_stratified_pc(df, xlim, ylim, stratify_key, this_save_path, cmap, flip)

# normalize across all PCs
df, cmap, vmin, vmax = normalize_intensities_and_get_colormap(df, pcts=[5, 95], cm_name="YlGnBu")

# %%
vmax

# %%
# save contrast adjusted reconstruction plots

use_vmin = vmin
use_vmax = vmax

for idx, _ in enumerate(fnames):
    fname = fnames[idx]
    df = pd.read_csv(f"{this_save_path}/{fname}", index_col=0)
    df[key] = names[idx]
    this_name = names[idx]
    df = normalize_intensities_and_get_colormap_apply(df, use_vmin, use_vmax)
    np_arr = df[["x", "y", "z"]].values
    colors = cmap(df["inorm"].values)[:, :3]
    np_arr2 = colors
    np_arr = np.concatenate([np_arr, np_arr2], axis=1)
    np.save(this_save_path / Path(f"{this_name}.npy"), np_arr)

# %% [markdown]
# # Pseudo time

# %%
# Compute pseudo time bins

bins = [
    (247.407, 390.752),
    (390.752, 533.383),
    (533.383, 676.015),
    (676.015, 818.646),
    (818.646, 961.277),
]
correct_bins = []
for ind, row in all_ret.iterrows():
    this_bin = []
    for bin_ in bins:
        if (row["volume_of_nucleus_um3"] > bin_[0]) and (row["volume_of_nucleus_um3"] <= bin_[1]):
            this_bin.append(bin_)
    if row["volume_of_nucleus_um3"] < bins[0][0]:
        this_bin.append(bin_)
    if row["volume_of_nucleus_um3"] > bins[4][1]:
        this_bin.append(bin_)
    assert len(this_bin) == 1
    correct_bins.append(this_bin[0])
all_ret["vol_bins"] = correct_bins
import pandas as pd

all_ret["vol_bins_inds"] = pd.factorize(all_ret["vol_bins"])[0]

# %%
all_ret = all_ret.groupby(["vol_bins"]).sample(n=75).reset_index(drop=True)

# %%
all_ret["cell_stage_fine"].value_counts()

# %%
z_max = 0.2
z_ind = 2
use_vmin = 5.03
use_vmax = 10

# %%
# Save reconstructions per bin

this_save_path = Path(save_path) / Path("pseudo_time_2")
this_save_path.mkdir(parents=True, exist_ok=True)

cols = [i for i in all_ret.columns if "mu" in i]
for ind, gr in all_ret.groupby(["vol_bins"]):
    this_stage_df = gr.reset_index(drop=True)
    this_stage_mu = this_stage_df[cols].values
    mean_mu = this_stage_mu.mean(axis=0)
    dist = (this_stage_mu - mean_mu) ** 2
    dist = np.sum(dist, axis=1)
    closest_idx = np.argmin(dist)
    real_input = this_stage_df.iloc[closest_idx]["CellId"]

    z_inf = torch.tensor(mean_mu).unsqueeze(axis=0)
    z_inf = z_inf.to(device)
    z_inf = z_inf.float()

    decoder = model.decoder["pcloud"]
    xhat = decoder(z_inf)
    xhat = save_pcloud(xhat[0], this_save_path, str(ind), z_max, z_ind)


names = os.listdir(this_save_path)
names = [i for i in names if i.split(".")[-1] == "csv"]
names = [i.split(".csv")[0] for i in names]
plot_pc_saved(this_save_path, names, key, flip, 0.5, views, xlim, ylim)

# %%
this_save_path

# %%
# Save contrast adjusted recons

items = os.listdir(this_save_path)
items = [this_save_path / Path(i) for i in items if i.split(".")[-1] == "csv"]

all_df = []
for j, i in enumerate(items):
    df = pd.read_csv(i)
    df["cluster"] = str(i).split("/")[-1][:-4]
    df = df.loc[df["z"] < 0.4]
    df = df.loc[df["z"] > -0.4].reset_index(drop=True)
    all_df.append(df)
df = pd.concat(all_df, axis=0).reset_index(drop=True)

for clust in df["cluster"].unique():
    df_2 = df.loc[df["cluster"] == clust].reset_index(drop=True)
    df_2 = normalize_intensities_and_get_colormap_apply(df_2, vmin=use_vmin, vmax=use_vmax)
    colors = cmap(df_2["inorm"].values)[:, :3]
    np_arr = df_2[["x", "y", "z"]].values
    np_arr2 = colors
    np_arr = np.concatenate([np_arr, np_arr2], axis=1)
    np.save(Path(this_save_path) / Path(f"{clust}.npy"), np_arr)

# %% [markdown]
# # Archetype analysis

# %%
# Save 8 archetypes
this_ret = all_ret
labels = this_ret["cell_stage_fine"].values
matrix = this_ret[[i for i in this_ret.columns if "mu" in i]].values

n_archetypes = 8
aa = AA_Fast(n_archetypes, max_iter=1000, tol=1e-6).fit(matrix)
archetypes_df = pd.DataFrame(aa.Z, columns=[f"mu_{i}" for i in range(matrix.shape[1])])

# %%
z_max = 0.2
z_ind = 2
use_vmin = 5.03
use_vmax = 10

# %%
# Save archetypes
this_save_path = Path(save_path) / Path("archetypes")
this_save_path.mkdir(parents=True, exist_ok=True)

model = model.eval()
key = "pcloud"
all_xhat = []
with torch.no_grad():
    for i in range(n_archetypes):
        z_inf = torch.tensor(archetypes_df.iloc[i].values).unsqueeze(axis=0)
        z_inf = z_inf.to(device)
        z_inf = z_inf.float()
        decoder = model.decoder[key]
        xhat = decoder(z_inf)
        xhat = xhat.detach().cpu().numpy()
        xhat = save_pcloud(xhat[0], this_save_path, i, z_max, z_ind)
        print(xhat.shape)
        all_xhat.append(xhat)

names = [str(i) for i in range(n_archetypes)]
plot_pc_saved(this_save_path, names, key, flip, 0.5, views, xlim, ylim)

# %%
key

# %%
# Save contrast adjusted numpy arrays
key = "archetype"
import os

items = os.listdir(this_save_path)
fnames = [i for i in items if i.split(".")[-1] == "csv"]
names = [i.split(".")[0] for i in fnames]

df = pd.DataFrame([])
for idx, _ in enumerate(fnames):
    fname = fnames[idx]
    print(fname)
    dft = pd.read_csv(f"{this_save_path}/{fname}", index_col=0)
    dft[key] = names[idx]
    df = pd.concat([df, dft], ignore_index=True)

archetypes = ["0", "1", "2", "3", "4", "5", "6", "7"]

for arch in archetypes:
    this_df = df.loc[df["archetype"] == arch].reset_index(drop=True)
    np_arr = this_df[["x", "y", "z"]].values
    this_df = normalize_intensities_and_get_colormap_apply(this_df, use_vmin, use_vmax)
    colors = cmap(this_df["inorm"].values)[:, :3]
    np_arr2 = colors
    np_arr = np.concatenate([np_arr, np_arr2], axis=1)
    print(np_arr.shape)
    np.save(this_save_path / Path(f"{arch}.npy"), np_arr)

# %%
use_vmax

# %%
