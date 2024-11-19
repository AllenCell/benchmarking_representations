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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pyntcloud import PyntCloud
from skimage import measure
from skimage.io import imread

from br.data.preprocessing.pc_preprocessing.pcna import (
    compute_labels as compute_labels_pcna,
)
from br.data.preprocessing.pc_preprocessing.punctate_cyto import (
    compute_labels as compute_labels_var_cyto,
)
from br.data.preprocessing.pc_preprocessing.punctate_nuc import (
    compute_labels as compute_labels_var_nuc,
)
from br.features.utils import normalize_intensities_and_get_colormap_apply


# %%
# utility plot functions
def plot_image(ax_array, struct, nuc, mem, vmin, vmax, num_slices=None, show_nuc_countour=True):
    mid_z = int(struct.shape[0] / 2)

    if num_slices is None:
        num_slices = mid_z * 2
    z_interp = np.linspace(mid_z - num_slices / 2, mid_z + num_slices / 2, num_slices + 1).astype(
        int
    )
    if z_interp.max() == struct.shape[0]:
        z_interp = z_interp[:-1]

    struct = np.where(mem, struct, 0)
    mem = mem[z_interp].max(0)
    nuc = nuc[z_interp].max(0)
    mem_contours = measure.find_contours(mem, 0.5)
    nuc_contours = measure.find_contours(nuc, 0.5)

    for ind, _ in enumerate(ax_array):
        this_struct = struct
        if ind > 0:
            this_struct = np.zeros(struct.shape)
        ax_array[ind].imshow(this_struct[z_interp].max(0), cmap="gray_r", vmin=vmin, vmax=vmax)
        if ind == 0:
            if show_nuc_countour:
                for contour in nuc_contours:
                    ax_array[ind].plot(contour[:, 1], contour[:, 0], linewidth=1, c="cyan")
            for contour in mem_contours:
                ax_array[ind].plot(contour[:, 1], contour[:, 0], linewidth=1, c="magenta")
        ax_array[ind].axis("off")
    return ax_array, z_interp


def plot_pointcloud(
    this_ax_array,
    points_all,
    z_interp,
    cmap,
    save_path=None,
    name=None,
    center=None,
    save=False,
    center_slice=False,
):
    this_p = points_all.loc[points_all["z"] < max(z_interp)]
    if center_slice:
        this_p = this_p.loc[this_p["z"] > min(z_interp)]
        print(this_p.shape)
    intensity = this_p.inorm.values
    this_ax_array.scatter(
        this_p["x"].values, this_p["y"].values, c=cmap(intensity), s=0.3, alpha=0.5
    )
    this_ax_array.axis("off")
    if save:
        z_center, y_center, x_center = center[0], center[1], center[2]

        # Center and scale for viz
        this_p["z"] = this_p["z"] - z_center
        this_p["y"] = this_p["y"] - y_center
        this_p["x"] = this_p["x"] - x_center

        this_p["z"] = 0.1 * this_p["z"]
        this_p["x"] = 0.1 * this_p["x"]
        this_p["y"] = 0.1 * this_p["y"]
        Path(save_path).mkdir(parents=True, exist_ok=True)
        colors = cmap(this_p["inorm"].values)[:, :3]
        np_arr = this_p[["x", "y", "z"]].values
        np_arr2 = colors
        np_arr = np.concatenate([np_arr, np_arr2], axis=1)
        np.save(Path(save_path) / Path(f"{name}.npy"), np_arr)


# %%
# Set paths
os.chdir("../../")
save_path = "./viz_variance_pointclouds2/"
# save_path = './viz_pcna_pointclouds/'

# %%
# PCNA
# PCNA_SINGLE_CELL_PROCESSED_PATH = ""
# ORIG_SINGLE_CELL_MANIFEST = ""
# orig_image_df = pd.read_parquet(PCNA_SINGLE_CELL_PROCESSED_PATH)
# df_all = pd.read_csv(ORIG_SINGLE_CELL_MANIFEST)
# orig_image_df = orig_image_df.merge(df_all[['CellId', 'crop_raw', 'crop_seg']], on='CellId')
# raw_ind = 2
# nuc_ind = 6
# mem_ind = 7

# Other punctate
# PUNCTATE_SINGLE_CELL_PROCESSED_PATH = ""
# ORIG_SINGLE_CELL_MANIFEST = ""
orig_image_df = pd.read_parquet(PUNCTATE_SINGLE_CELL_PROCESSED_PATH)
df_full = pd.read_csv(ORIG_SINGLE_CELL_MANIFEST)
orig_image_df = orig_image_df.merge(df_full[["CellId", "crop_seg"]], on="CellId")
raw_ind = 2
nuc_ind = 3
mem_ind = 4

# for nuc structures
df = pd.read_parquet(PUNCTATE_SINGLE_CELL_PROCESSED_PATH)
orig_image_df = orig_image_df.merge(df[["registered_path", "CellId"]], on="CellId")

# %%
# Sample CellId
# strat = 'cell_stage_fine'
# strat_val = 'lateS-G2'

strat = "Structure"
strat_val = "SON"
this_image = orig_image_df.loc[orig_image_df[strat] == strat_val].sample(n=1)
# this_image = orig_image_df.loc[orig_image_df['CellId'] == 'c6b66235-554c-4fd3-b0a2-a1e5468afb64']
cell_id = this_image["CellId"].iloc[0]
strat_val = this_image[strat].iloc[0]

# %%
strat_val

# %%
# Sample dense point cloud and get images

# points_all, struct, img, center = compute_labels_pcna(this_image.iloc[0], False)
points_all, struct, img, center = compute_labels_var_nuc(this_image.iloc[0], False)
# points_all, struct, img, center = compute_labels_var_cyto(this_image.iloc[0], False)
img_raw = img[raw_ind]
img_nuc = img[nuc_ind]
img_mem = img[mem_ind]
img_raw = np.where(img_raw < 60000, img_raw, img_raw.min())

# from saved PC
# points_all = PyntCloud.from_file(this_image['pcloud_path_updated_morepoints'].iloc[0]).points
# z_center, y_center, x_center = center[0], center[1], center[2]
# # Center and scale for viz
# points_all["z"] = points_all["z"] + z_center
# points_all["y"] = points_all["y"] + y_center
# points_all["x"] = points_all["x"] + x_center

# points_saved = PyntCloud.from_file(this_image['pcloud_path_structure_norm'].iloc[0]).points
# z_center, y_center, x_center = center[0], center[1], center[2]
# # Center and scale for viz
# points_saved["z"] = points_saved["z"] + z_center
# points_saved["y"] = points_saved["y"] + y_center
# points_saved["x"] = points_saved["x"] + x_center

# %%
# Sample sparse point cloud and get images

probs2 = points_all["s"].values
probs2 = np.where(probs2 < 0, 0, probs2)
probs2 = probs2 / probs2.sum()
idxs2 = np.random.choice(np.arange(len(probs2)), size=2048, replace=True, p=probs2)
points = points_all.iloc[idxs2].reset_index(drop=True)

# %%
# Apply contrast to point clouds

# for perox 415, 515
# for endos 440, 600
vmin = 420
vmax = 600
points = normalize_intensities_and_get_colormap_apply(points, vmin, vmax)
points_all = normalize_intensities_and_get_colormap_apply(points_all, vmin, vmax)

# %%
# %matplotlib inline

save = True
center_slice = False

fig, ax_array = plt.subplots(1, 3, figsize=(10, 5))

ax_array, z_interp = plot_image(
    ax_array, img_raw, img_nuc, img_mem, vmin, vmax, num_slices=15, show_nuc_countour=True
)
ax_array[0].set_title("Raw image")

name = strat_val + "_" + str(cell_id)

plot_pointcloud(
    ax_array[1],
    points_all,
    z_interp,
    plt.get_cmap("YlGnBu"),
    save_path=save_path,
    name=name,
    center=center,
    save=False,
    center_slice=center_slice,
)
ax_array[1].set_title("Sampling dense PC")

plot_pointcloud(
    ax_array[2],
    points,
    z_interp,
    plt.get_cmap("YlGnBu"),
    save_path=save_path,
    name=name,
    center=center,
    save=save,
    center_slice=center_slice,
)
ax_array[2].set_title("Sampling sparse PC")

# plt.show()
fig.savefig(save_path + f"/{name}.png", bbox_inches="tight", dpi=300)

# %%
# add scalebar
# df = orig_image_df
# from aicsimageio import AICSImage
# from matplotlib_scalebar.scalebar import ScaleBar
# selected_cellids=['c6b66235-554c-4fd3-b0a2-a1e5468afb64']
# sub_slice_list = ['PCNA']
# padding=10
# imgs = []
# for i, cell_id in enumerate(selected_cellids):
#     structure=sub_slice_list[i]
#     img_path = df.loc[df['CellId'] == cell_id, 'crop_raw'].values[0]
#     seg_path = df.loc[df['CellId'] == cell_id, 'crop_seg'].values[0]
#     img = AICSImage(img_path).data.squeeze()[-1]
#     mem = AICSImage(seg_path).data.squeeze()[1]

#     mem = mem[z_interp].max(0)
#     mem_contours = measure.find_contours(mem, 0.5)

#     if structure in ['HIST1H2BJ','NUP153','SMC1A']:
#         seg_idx = 0
#     else:
#         seg_idx = 1
#     seg = AICSImage(seg_path).data.squeeze()[seg_idx]
#     binary_mask = seg.astype(bool)
#     background_value = np.median(img)
#     masked_img = np.full_like(img, fill_value=background_value)
#     masked_img[binary_mask] = img[binary_mask]
#     if structure == 'NUP153':
#         slice = masked_img.shape[0] // 2
#         displ_img = masked_img[slice,:,:]
#     else:
#         displ_img = masked_img.max(0)
#     rows = np.any(displ_img, axis=1)
#     cols = np.any(displ_img, axis=0)
#     rmin, rmax = np.where(rows)[0][[0, -1]]
#     cmin, cmax = np.where(cols)[0][[0, -1]]
#     rmin = max(rmin - padding, 0)
#     rmax = min(rmax + padding, displ_img.shape[0])
#     cmin = max(cmin - padding, 0)
#     cmax = min(cmax + padding, displ_img.shape[1])
#     res_img = displ_img[rmin:rmax, cmin:cmax]
#     imgs.append(res_img)

# # scale_formatter = lambda value, unit: f""

# max_ht, max_wid = imgs[0].shape
# for i, cell_id in enumerate(selected_cellids, start=1):
#     structure=sub_slice_list[i-1]
#     img = imgs[i-1]
#     background_value = np.median(img)

#     pad_height = max(max_ht - img.shape[0], 0)
#     pad_width = max(max_wid - img.shape[1], 0)

#     pad_img = np.pad(img,
#                           pad_width=((pad_height//2, pad_height - pad_height//2),
#                                      (pad_width//2, pad_width - pad_width//2)),
#                           mode='constant',
#                           constant_values=background_value)

#     fig, ax = plt.subplots(1,1, figsize=(8, 5))
#     ax.imshow(pad_img, cmap='gray_r')
#     for contour in mem_contours:
#         ax.plot(contour[:, 1], contour[:, 0], linewidth=1, c='magenta')
#     ax.set_title(f'CellId: {cell_id}; {structure}')
#     ax.axis('off')
#     scalebar = ScaleBar(0.108333, 'um', length_fraction=0.25,
#                     location='upper right',
#                     frameon=True,
#                     color='black',
#                     scale_loc='bottom',
#                     box_color='white',
#                     box_alpha=1)
#                        #scale_formatter=scale_formatter)
#     ax.add_artist(scalebar)

#     fig.savefig(save_path + 'mids-lates_scalaebar.png', bbox_inches='tight', dpi=300)

# %%
