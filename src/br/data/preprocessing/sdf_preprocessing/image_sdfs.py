import numpy as np
import pandas as pd
import pyvista as pv
from aicsimageio import AICSImage
from monai.transforms import FillHoles
from br.data.utils import (
    get_mesh_from_image,
    get_scaled_mesh,
    get_sdf_from_mesh_vtk,
    voxelize_scaled_mesh
)
from tqdm import tqdm
from multiprocessing import Pool

path = './morphology_appropriate_representation_learning/preprocessed_data/npm1/manifest.csv'
scale_factor_path = './morphology_appropriate_representation_learning/preprocessed_data/npm1/scale_factor.npz'
out_dir_scaled_seg = "./outputs_seg_npm1/"
out_dir_scaled_sdf = "./outputs_sdf_npm1/"
out_dir_mesh = "./outputs_mesh_npm1/"

sc_factor_data = np.load(scale_factor_path, allow_pickle=True)
scale_factor_dict = dict(zip(sc_factor_data["keys"], sc_factor_data["values"]))

df = pd.read_csv(path)


test_ids = [964798, 661110, 644401, 967887, 703621, 644479]
df = df.loc[df['CellId'].isin(test_ids)].reset_index(drop=True)
hole_fill_transform = FillHoles()

all_rows = []
for ind, row in tqdm(df.iterrows(), total=len(df)):
    all_rows.append(row)

def process(r):
    try:
        cellid = r["CellId"]

        out_path_sdf = f"{out_dir_scaled_sdf}/{cellid}"
        out_path_seg = f"{out_dir_scaled_seg}/{cellid}"
        out_path_mesh = f"{out_dir_mesh}/{cellid}.stl"

        seg = AICSImage(r["crop_seg_masked"]).data.squeeze()
        seg = hole_fill_transform(seg).numpy()

        # print(seg.shape)

        mesh, _, _ = get_mesh_from_image(seg, sigma=0, lcc=False, denoise=False)

        pv.wrap(mesh).save(out_path_mesh)
        vox_resolution = 64
        # vox_resolution = 32

        sdf, scale_factor = get_sdf_from_mesh_vtk(
            None, vox_resolution=vox_resolution, scale_factor=global_scale_factor_64_res, vpolydata=mesh
        )
        np.save(out_path_sdf, sdf)
        vox_shape = (vox_resolution, vox_resolution, vox_resolution)
        scaled_mesh, _ = get_scaled_mesh(
            None, int(vox_resolution), global_scale_factor_64_res, mesh, True
        )

        # coords = numpy_support.vtk_to_numpy(scaled_mesh.GetPoints().GetData())
        # centroid = coords.mean(axis=0, keepdims=True)
        # print(centroid)

        scaled_seg = voxelize_scaled_mesh(scaled_mesh)
        com = scaled_seg.shape
        print(com)
        pad = []
        for i, j in zip(vox_shape, com):
            pad.append((int(i - j)//2, int(i - j)//2))
        scaled_seg = np.pad(scaled_seg, pad)
        # scaled_seg = get_image_from_mesh(scaled_mesh, vox_shape, 0)

        np.save(out_path_seg, scaled_seg)
    except:
        print(r['CellId'])

# with Pool(40) as p:
with Pool(1) as p:
    _ = tuple(
        tqdm(
            p.imap_unordered(
                process,
                all_rows,
            ),
            total=len(all_rows),
            desc="compute_everything",
        )
    )


# for i, r in tqdm(df.iterrows(), total=len(df)):

