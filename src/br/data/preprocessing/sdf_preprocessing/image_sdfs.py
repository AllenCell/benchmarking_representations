import numpy as np
import pandas as pd
import pyvista as pv
from aicsimageio import AICSImage
from monai.transforms import FillHoles

from br.data.utils import (
    get_mesh_from_image,
    get_scaled_seg_from_mesh,
    get_sdf_from_mesh,
)

path = PATH_TO_SINGLE_CELL_DATA

parent_out_dir = "..."
path = ""
out_dir_scaled_seg = "./"
out_dir_scaled_sdf = "./"
out_dir_mesh = "./"
df = pd.read_csv(path)

hole_fill_transform = FillHoles()

for i, r in df.iterrows():
    cellid = r["CellId"]

    out_path_sdf = f"{out_dir_scaled_sdf}/{cellid}"
    out_path_seg = f"{out_dir_scaled_seg}/{cellid}"
    out_path_mesh = f"{out_dir_mesh}/{cellid}.stl"

    seg = AICSImage(r["crop_seg"]).data.squeeze()
    seg = hole_fill_transform(seg).numpy()

    mesh, _, _ = get_mesh_from_image(seg, sigma=0, lcc=False, denoise=False)

    pv.wrap(mesh).save(out_path_mesh)

    sdf, scale_factor = get_sdf_from_mesh(
        path=None, vox_resolution=32, scale_factor=None, vpolydata=mesh
    )
    np.save(out_path_sdf, sdf)

    scaled_seg, _ = get_scaled_seg_from_mesh(
        path=None, vox_resolution=32, scale_factor=scale_factor, vpolydata=mesh
    )
    np.save(out_path_seg, scaled_seg)
