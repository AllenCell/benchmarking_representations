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
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-e23853d6-1ca4-59e9-ac9a-1887267908f3"
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from hydra.utils import instantiate
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset

from br.features.rotation_invariance import rotation_image_batch_z, rotation_pc_batch_z
from br.models.load_models import load_model_from_path

device = "cuda:0"

# %%
# Set paths
os.chdir("/allen/aics/modeling/ritvik/projects/benchmarking_representations/")
save_path = "./test_cellpack_recons/"
results_path = "./configs/results/"

# %%
# Load data yaml and test batch
cellpack_data = "./configs/data/cellpack/pc.yaml"
with open(cellpack_data) as stream:
    cellpack_data = yaml.safe_load(stream)
data = instantiate(cellpack_data)
batch = next(iter(data.test_dataloader()))

# %% [markdown]
# # Save examples of raw data

# %%
from pathlib import Path

this_save_path = Path(save_path) / Path("panel_a")
this_save_path.mkdir(parents=True, exist_ok=True)

all_arr = []
for i in range(6):
    np_arr = batch["pcloud"][i].numpy()
    new_array = np.zeros(np_arr.shape)
    z = np_arr[:, 0]
    # inds = np.where(z > 0.1)[0]
    new_array[:, 0] = np_arr[:, 2]
    new_array[:, 1] = z
    new_array[:, 2] = np_arr[:, 1]
    new_array = new_array[inds]
    all_arr.append(new_array)
    np.save(this_save_path / Path(f"{i}.npy"), new_array)


# %% [markdown]
# # Visualize reconstructions and rotation invariance 

# %%
# utility function for plotting
def plot_pc(this_p, axes, max_size, color="gray", x_ind=2, y_ind=1):
    axes.scatter(this_p[:, x_ind], this_p[:, y_ind], c=color, s=1)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["bottom"].set_visible(False)
    axes.spines["left"].set_visible(False)
    axes.set_aspect("equal", adjustable="box")
    axes.set_ylim([-max_size, max_size])
    axes.set_xlim([-max_size, max_size])
    axes.set_yticks([])
    axes.set_xticks([])


# %%
models, names, sizes = load_model_from_path("cellpack", results_path)

# %%
names

# %%
model = models[-3]
this_name = names[-3]

# %%
for key in batch.keys():
    if key not in [
        "split",
        "bf_meta_dict",
        "egfp_meta_dict",
        "filenames",
        "image_meta_dict",
        "cell_id",
    ]:
        if not isinstance(batch[key], list):
            batch[key] = batch[key].to(device)

# %%
this_save_path = Path(save_path) / Path(f"Recons_{this_name}")
this_save_path.mkdir(parents=True, exist_ok=True)

this_key = "pcloud"

max_z = {0: 20, 1: 20, 2: 20, 3: 1, 4: 20, 5: 20}
max_size = 10

all_thetas = [
    0,
    1 * 90,
    2 * 90,
    3 * 90,
]


all_xhat = []
all_canon = []
all_input = []
with torch.no_grad():
    for jl, theta in enumerate(all_thetas):
        this_input_rot = rotation_pc_batch_z(
            batch,
            theta,
        )
        batch_input = {this_key: torch.tensor(this_input_rot).to(device).float()}
        z, z_params = model.get_embeddings(batch_input, inference=True)
        xhat = model.decode_embeddings(z_params, batch_input, decode=True, return_canonical=True)
        all_input.append(this_input_rot)
        if theta == 0:
            for ind in range(6):
                this_p = this_input_rot[ind]
                this_max_z = max_z[ind]
                this_p = this_p[np.where(this_p[:, 0] < this_max_z)[0]]
                this_p = this_p[np.where(this_p[:, 0] > -this_max_z)[0]]
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                plot_pc(this_p, ax, max_size, "black")
                fig.savefig(this_save_path / f"input_{ind}.png")

        if "canonical" in xhat.keys():
            this_canon = xhat["canonical"].detach().cpu().numpy()
            all_canon.append(this_canon)
            if theta == 0:
                for ind in range(6):
                    this_p = this_canon[ind]
                    this_max_z = max_z[ind]
                    this_p = this_p[np.where(this_p[:, 1] < this_max_z)[0]]
                    this_p = this_p[np.where(this_p[:, 1] > -this_max_z)[0]]
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    plot_pc(this_p, ax, max_size, "black", x_ind=2, y_ind=1)
                    fig.savefig(this_save_path / f"canon_{ind}.png")
        this_recon = xhat[this_key].detach().cpu().numpy()
        all_xhat.append(this_recon)
        if theta == 0:
            for ind in range(6):
                this_p = this_recon[ind]
                this_max_z = max_z[ind]
                this_p = this_p[np.where(this_p[:, 0] < this_max_z)[0]]
                this_p = this_p[np.where(this_p[:, 0] > -this_max_z)[0]]
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                plot_pc(this_p, ax, max_size, "black")
                fig.savefig(this_save_path / f"recon_{ind}.png")

# %%
all_input[0][0].max(axis=0)

# %%
i = 0  # rot ind
ind = 0  # rule
max_z = 1
max_size = 10

# this_p = all_input[i][ind].detach().cpu().numpy()
this_p = all_xhat[i][ind]
this_p = this_p[np.where(this_p[:, 0] < max_z)[0]]
this_p = this_p[np.where(this_p[:, 0] > -max_z)[0]]
print(this_p.max(axis=0))
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plot_pc(this_p, ax, max_size, "black")

# %%
fig, axes = plt.subplots(1, 4, figsize=(10, 5))


ind = 0
max_z = 200
max_size = 10
for i in range(4):
    this_p = all_input[i][ind].detach().cpu().numpy()
    this_p = this_p[np.where(this_p[:, 1] < max_z)[0]]
    this_p = this_p[np.where(this_p[:, 1] > -max_z)[0]]
    print(this_p.shape)
    plot_pc(this_p, axes[i], max_size)

# fig.savefig('./cellpack_rot_test/7abfecf1-44db-468a-b799-4959a23cfb0d_pc_rot_input.png', dpi=300, bbox_inches='tight')

fig, axes = plt.subplots(1, 4, figsize=(10, 5))
for i in range(4):
    this_p = all_canon[i][ind]
    this_p = this_p[np.where(this_p[:, 1] < max_z)[0]]
    this_p = this_p[np.where(this_p[:, 1] > -max_z)[0]]
    plot_pc(this_p, axes[i], max_size)

# fig.savefig('./cellpack_rot_test/7abfecf1-44db-468a-b799-4959a23cfb0d_pc_rot_canon.png', dpi=300, bbox_inches='tight')
fig, axes = plt.subplots(1, 4, figsize=(10, 5))
for i in range(4):
    this_p = all_xhat[i][ind]
    this_p = this_p[np.where(this_p[:, 1] < max_z)[0]]
    this_p = this_p[np.where(this_p[:, 1] > -max_z)[0]]
    plot_pc(this_p, axes[i], max_size)

# fig.savefig('./cellpack_rot_test/7abfecf1-44db-468a-b799-4959a23cfb0d_pc_rot_recon_classical.png', dpi=300, bbox_inches='tight')

# %%

# %%

# %%

# %%

# %%

# %%
import pandas as pd

gg = pd.read_csv("/allen/aics/modeling/ritvik/forSaurabh/manifest.csv")

# %%
path = gg.loc[gg["CellId"] == "9c1ff213-4e9e-4b73-a942-3baf9d37a50f"]["nucobj_path"].iloc[0]

# %%
this_save_path

# %%
# %load_ext autoreload
# %autoreload 2

mi.set_variant("scalar_rgb")
import os

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import trimesh
from mitsuba import ScalarTransform4f as T
from trimesh.transformations import rotation_matrix


def plot(this_mesh_path, angle, angle2=None, angle3=None, name=None):
    myMesh = trimesh.load(this_mesh_path)

    # Scale the mesh to approximately one unit based on the height
    sf = 1.0
    myMesh.apply_scale(sf / myMesh.extents.max())

    # for 3_1
    myMesh = myMesh.apply_transform(rotation_matrix(np.deg2rad(angle), [0, 0, -1]))
    if angle2:
        myMesh = myMesh.apply_transform(rotation_matrix(np.deg2rad(angle2), [0, 1, 0]))

    if angle3:
        myMesh = myMesh.apply_transform(rotation_matrix(np.deg2rad(angle3), [1, 0, 0]))
    # myMesh = myMesh.apply_transform(rotation_matrix(np.deg2rad(0), [1,0,0]))

    # Translate the mesh so that it's centroid is at the origin and rests on the ground plane
    myMesh.apply_translation(
        [
            -myMesh.bounds[0, 0] - myMesh.extents[0] / 2.0,
            -myMesh.bounds[0, 1] - myMesh.extents[1] / 2.0,
            -myMesh.bounds[0, 2],
        ]
    )

    # Fix the mesh normals for the mesh
    myMesh.fix_normals()

    # Write the mesh to an external file (Wavefront .obj)
    with open("mesh.obj", "w") as f:
        f.write(trimesh.exchange.export.export_obj(myMesh, include_normals=True))

    # Create a sensor that is used for rendering the scene
    def load_sensor(r, phi, theta):
        # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
        origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

        return mi.load_dict(
            {
                "type": "perspective",
                "fov": 15.0,
                "to_world": T.look_at(
                    origin=origin, target=[0, 0, myMesh.extents[2] / 2], up=[0, 0, 1]
                ),
                "sampler": {"type": "independent", "sample_count": 16},
                "film": {
                    "type": "hdrfilm",
                    "width": 1024,
                    "height": 768,
                    "rfilter": {
                        "type": "tent",
                    },
                    "pixel_format": "rgb",
                },
            }
        )

    # Scene parameters
    relativeLightHeight = 8

    # A scene dictionary contains the description of the rendering scene.
    scene2 = mi.load_dict(
        {
            "type": "scene",
            # The keys below correspond to object IDs and can be chosen arbitrarily
            "integrator": {"type": "path"},
            "mesh": {
                "type": "obj",
                "filename": "mesh.obj",
                "face_normals": True,  # This prevents smoothing of sharp-corners by discarding surface-normals. Useful for engineering CAD.
                "bsdf": {
                    # 'type': 'diffuse',
                    # 'reflectance': {
                    #     'type': 'rgb',
                    #     'value': [0.1, 0.27, 0.86]
                    # }
                    # 'type': 'plastic',
                    # 'diffuse_reflectance': {
                    #     'type': 'rgb',
                    #     'value': [0.1, 0.27, 0.36]
                    # },
                    # 'int_ior': 1.9
                    # 'type': 'roughplastic'
                    "type": "pplastic",
                    "diffuse_reflectance": {"type": "rgb", "value": [0.05, 0.03, 0.1]},
                    "alpha": 0.02,
                },
            },
            # A general emitter is used for illuminating the entire scene (renders the background white)
            "light": {"type": "constant", "radiance": 1.0},
            "areaLight": {
                "type": "rectangle",
                # The height of the light can be adjusted below
                "to_world": T.translate([0, 0.0, myMesh.bounds[1, 2] + relativeLightHeight])
                .scale(1.0)
                .rotate([1, 0, 0], 5.0),
                "flip_normals": True,
                "emitter": {
                    "type": "area",
                    "radiance": {
                        "type": "spectrum",
                        "value": 30.0,
                    },
                },
            },
            "floor": {
                "type": "disk",
                "to_world": T.scale(3).translate([0.0, 0.0, 0.0]),
                "material": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": 0.75},
                },
            },
        }
    )

    sensor_count = 1

    radius = 4
    phis = [130.0]
    theta = 60.0

    sensors = [load_sensor(radius, phi, theta) for phi in phis]

    """
    Render the Scene
    The render samples are specified in spp
    """
    image = mi.render(scene2, sensor=sensors[0], spp=256)

    # Write the output

    save_path = this_save_path
    mi.util.write_bitmap(str(save_path) + f"{name}.png", image)
    # mi.util.write_bitmap(save_path + ".exr", image)

    # Display the output in an Image
    plt.imshow(image ** (1.0 / 2.2))
    plt.axis("off")


# %%
path

# %%
plot(path, 0, 90, 0, "nuc")

# %%

# %%
aa = np.load(
    "/allen/aics/modeling/ritvik/projects/benchmarking_representations/viz_pcna_pointclouds/midS-lateS_c6b66235-554c-4fd3-b0a2-a1e5468afb64.npy"
)

# %%
aa.max(axis=0)

# %%
bb = np.load(
    "/allen/aics/modeling/ritvik/projects/benchmarking_representations/notebooks_old/variance_all_punctate/pcna/latent_walks/viz/midS-lateS_0_1.npy"
)

# %%
bb.max(axis=0)

# %%
aa = np.load(
    "/allen/aics/modeling/ritvik/projects/benchmarking_representations/viz_variance_pointclouds2/NUP153_692417.npy"
)

# %%
aa.max(axis=0)

# %%
bb = np.load(
    "/allen/aics/modeling/ritvik/projects/benchmarking_representations/test_var_punctate_embeddings/latent_walks/structure_name_NUP153_0_0.npy"
)

# %%
bb.max(axis=0)

# %%
