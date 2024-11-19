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
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-864c07c4-8eeb-5b23-8d57-eaeb942a9a0f"
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

# %%
os.chdir("../../benchmarking_representations/")
save_path = "./test_npm1_save_embeddings/"

# %%
dataset_name = "npm1"
batch_size = 2
debug = False
results_path = "./configs/results/"
data_list, all_models, run_names, model_sizes = get_data_and_models(
    dataset_name, batch_size, results_path, debug
)

# %% [markdown]
# # Compute embeddings and emissions

# %%
from br.models.save_embeddings import save_embeddings

splits_list = ["train", "val", "test"]
meta_key = None
eval_scaled_img = [False] * 5

gt_mesh_dir = MESH_DIR
gt_sampled_pts_dir = SAMPLE_DIR
gt_scale_factor_dict_path = SCALE_FACTOR_DIR

eval_scaled_img_params = [
    {
        "eval_scaled_img_model_type": "iae",
        "eval_scaled_img_resolution": 32,
        "gt_mesh_dir": gt_mesh_dir,
        "gt_scale_factor_dict_path": None,
        "gt_sampled_pts_dir": gt_sampled_pts_dir,
        "mesh_ext": "stl",
    },
    {
        "eval_scaled_img_model_type": "sdf",
        "eval_scaled_img_resolution": 32,
        "gt_mesh_dir": gt_mesh_dir,
        "gt_scale_factor_dict_path": gt_scale_factor_dict_path,
        "gt_sampled_pts_dir": None,
        "mesh_ext": "stl",
    },
    {
        "eval_scaled_img_model_type": "seg",
        "eval_scaled_img_resolution": 32,
        "gt_mesh_dir": gt_mesh_dir,
        "gt_scale_factor_dict_path": gt_scale_factor_dict_path,
        "gt_sampled_pts_dir": None,
        "mesh_ext": "stl",
    },
    {
        "eval_scaled_img_model_type": "sdf",
        "eval_scaled_img_resolution": 32,
        "gt_mesh_dir": gt_mesh_dir,
        "gt_scale_factor_dict_path": gt_scale_factor_dict_path,
        "gt_sampled_pts_dir": None,
        "mesh_ext": "stl",
    },
    {
        "eval_scaled_img_model_type": "seg",
        "eval_scaled_img_resolution": 32,
        "gt_mesh_dir": gt_mesh_dir,
        "gt_scale_factor_dict_path": gt_scale_factor_dict_path,
        "gt_sampled_pts_dir": None,
        "mesh_ext": "stl",
    },
]
loss_eval_list = [torch.nn.MSELoss(reduction="none")] * 5
sample_points_list = [False] * 5
skew_scale = None
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
run_names

# %% [markdown]
# # Latent walks

# %%
# Load model and embeddings

run_names = ["Rotation_invariant_pointcloud_SDF"]
DATASET_INFO = get_all_configs_per_dataset(results_path)
all_ret, df = get_embeddings(run_names, dataset_name, DATASET_INFO, save_path)
model = all_models[0]

# %%
save_path

# %%
import pyvista as pv
from cyto_dl.image.transforms import RotationMask
from skimage.io import imread
from sklearn.decomposition import PCA
from tqdm import tqdm

from br.data.utils import mesh_seg_model_output
from br.visualization.mitsuba_render_image import plot

this_save_path = Path(save_path) / Path("latent_walks")
this_save_path.mkdir(parents=True, exist_ok=True)

lw_dict = {"num_pieces": [], "PC": [], "bin": [], "CellId": []}
for num_pieces in all_ret["STR_connectivity_cc_thresh"].unique():
    this_sub_m = all_ret.loc[all_ret["STR_connectivity_cc_thresh"] == num_pieces].reset_index(
        drop=True
    )
    all_features = this_sub_m[[i for i in this_sub_m.columns if "mu" in i]].values
    latent_dim = 512
    dim_size = latent_dim
    x_label = "pcloud"
    pca = PCA(n_components=dim_size)
    pca_features = pca.fit_transform(all_features)
    pca_std_list = pca_features.std(axis=0)
    for rank in [0, 1]:
        all_xhat = []
        all_closest_real = []
        all_closest_img = []
        latent_walk_range = [-2, 0, 2]
        for value_index, value in enumerate(tqdm(latent_walk_range, total=len(latent_walk_range))):
            z_inf = torch.zeros(1, dim_size)
            z_inf[:, rank] += value * pca_std_list[rank]
            z_inf = pca.inverse_transform(z_inf).numpy()

            dist = (all_features - z_inf) ** 2
            dist = np.sum(dist, axis=1)
            closest_idx = np.argmin(dist)
            closest_real_id = this_sub_m.iloc[closest_idx]["CellId"]
            mesh = pv.read(
                all_ret.loc[all_ret["CellId"] == closest_real_id]["mesh_path_noalign"].iloc[0]
            )
            mesh.save(this_save_path / Path(f"{num_pieces}_{rank}_{value_index}.ply"))

            lw_dict["num_pieces"].append(num_pieces)
            lw_dict["PC"].append(rank)
            lw_dict["bin"].append(value_index)
            lw_dict["CellId"].append(closest_real_id)

            # this_mesh_path = this_save_path / Path(f'{num_pieces}_{rank}_{value_index}.ply')
            # this_mesh_path = './' + str(this_mesh_path)

            # mitsuba_save_path = this_save_path / Path('mitsuba')
            # mitsuba_save_path.mkdir(parents=True, exist_ok=True)
            # mitsuba_save_path = './' + str(mitsuba_save_path)
            # name = f"{num_pieces}_{rank}_{value_index}"

            # plot(str(this_mesh_path), mitsuba_save_path, 120, None, None, name)

# %%
lw_dict = pd.DataFrame(lw_dict)
lw_dict.to_csv(this_save_path / "latent_walk.csv")

# %%
lw_dict

# %%
save_path = "./test_npm1_save_embeddings/"
this_save_path = Path(save_path) / Path("latent_walks")

# num_pieces = 4.0
num_pieces = "2.0"
rank = 0
bin_ = 0
this_mesh_path = this_save_path / Path(f"{num_pieces}_{rank}_{bin_}.ply")
this_mesh_path = "./" + str(this_mesh_path)

save_path = this_save_path / Path("mitsuba")
save_path.mkdir(parents=True, exist_ok=True)
save_path = "./" + str(save_path)
name = f"{num_pieces}_{rank}_{bin_}"


plot(str(this_mesh_path), save_path, 10, 0, None, name)

# %% [markdown]
# # Archetype

# %%
from br.features.archetype import AA_Fast

n_archetypes = 5
matrix = all_ret[[i for i in all_ret.columns if "mu" in i]].values
aa = AA_Fast(n_archetypes, max_iter=1000, tol=1e-6).fit(matrix)

import pandas as pd

archetypes_df = pd.DataFrame(aa.Z, columns=[f"mu_{i}" for i in range(matrix.shape[1])])

# %%
this_save_path = Path(save_path) / Path("archetypes")
this_save_path.mkdir(parents=True, exist_ok=True)

arch_dict = {"CellId": [], "archetype": []}
all_features = matrix
for i in range(n_archetypes):
    this_mu = archetypes_df.iloc[i].values
    dist = (all_features - this_mu) ** 2
    dist = np.sum(dist, axis=1)
    closest_idx = np.argmin(dist)
    closest_real_id = all_ret.iloc[closest_idx]["CellId"]
    print(dist, closest_real_id)
    mesh = pv.read(all_ret.loc[all_ret["CellId"] == closest_real_id]["mesh_path_noalign"].iloc[0])
    mesh.save(this_save_path / Path(f"{i}.ply"))
    arch_dict["archetype"].append(i)
    arch_dict["CellId"].append(closest_real_id)

# %%
arch_dict = pd.DataFrame(arch_dict)
arch_dict.to_csv(this_save_path / "archetypes.csv")

# %%
this_save_path

# %%
save_path = "./test_npm1_save_embeddings/"
this_save_path = Path(save_path) / Path("archetypes")

arch = "4"
this_mesh_path = this_save_path / Path(f"{arch}.ply")
this_mesh_path = "./" + str(this_mesh_path)

save_path = this_save_path / Path("mitsuba")
save_path.mkdir(parents=True, exist_ok=True)
save_path = "./" + str(save_path)
name = f"{arch}"


plot(str(this_mesh_path), save_path, 90, 0, None, name)

# %% [markdown]
# # Pseudo time

# %%
all_ret["volume_of_nucleus_um3"] = all_ret["dna_shape_volume_lcc"] * 0.108**3

# %%
feat = "volume_of_nucleus_um3"
upper = np.quantile(all_ret[feat], q=0.99)
lower = np.quantile(all_ret[feat], q=0.01)

this = all_ret.loc[all_ret[feat] < upper]
this = this.loc[this[feat] > lower].reset_index(drop=True)

# %%
this["vol_bins"] = pd.cut(this[feat], bins=5)
this["vol_bins_ind"] = pd.factorize(this["vol_bins"])[0]

# %%
this["vol_bins"].value_counts()

# %%
this_save_path = Path(save_path) / Path("pseudo_time")
this_save_path.mkdir(parents=True, exist_ok=True)

# %%
all_features = this[[i for i in this.columns if "mu" in i]].values

vol_dict = {"vol_bin": [], "CellId": []}
this["vol_bins"] = this["vol_bins"].astype(str)
for hh in this["vol_bins"].unique():
    this_ret = this.loc[this["vol_bins"] == hh].reset_index(drop=True)

    this_mu = np.expand_dims(
        this_ret[[i for i in this_ret.columns if "mu" in i]].mean(axis=0), axis=0
    )
    dist = (all_features - this_mu) ** 2
    # dist = np.sum(dist, axis=1)
    k = 1
    # print(min(latent_dim, all_features.shape[0]))
    inds = np.argpartition(dist.sum(axis=-1), k)[:k]  # get 10 closest
    closest_samples = this.iloc[inds].reset_index(drop=True)
    for ind, row in closest_samples.iterrows():
        # closest_real_id = this.iloc[closest_idx]['CellId']
        closest_real_id = row["CellId"]
        print(
            closest_idx,
            this_ret["vol_bins"].unique(),
            all_features.shape,
            this_ret.shape,
            this_ret["dna_shape_volume_lcc"].mean(),
            closest_real_id,
        )
        mesh = pv.read(
            all_ret.loc[all_ret["CellId"] == closest_real_id]["mesh_path_noalign"].iloc[0]
        )
        mesh.save(this_save_path / Path(f"{hh}_{ind}_{closest_real_id}.ply"))

        vol_dict["vol_bin"].append(hh)
        vol_dict["CellId"].append(closest_real_id)

# %%
vol_dict = pd.DataFrame(vol_dict)
vol_dict.to_csv(this_save_path / "pseudo_time.csv")

# %%
save_path = "./test_npm1_save_embeddings/"
this_save_path = Path(save_path) / Path("pseudo_time")

pseu = "(533.383, 676.015]_0_970952"
this_mesh_path = this_save_path / Path(f"{pseu}.ply")
this_mesh_path = "./" + str(this_mesh_path)

save_path = this_save_path / Path("mitsuba")
save_path.mkdir(parents=True, exist_ok=True)
save_path = "./" + str(save_path)
name = f"{pseu}"


plot(str(this_mesh_path), save_path, 90, 90, None, name)

# %%
