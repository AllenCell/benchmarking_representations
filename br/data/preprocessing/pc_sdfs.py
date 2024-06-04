import trimesh
import mesh_to_sdf
import os
import numpy as np
import pandas as pd

os.environ["PYOPENGL_PLATFORM"] = "egl"


def sample_points(mesh, n_points, p_type=np.float16):
    pcl, idx = mesh.sample(n_points, return_index=True)
    normals = mesh.face_normals[idx]

    out_dict = {
        "points": pcl.astype(p_type),
        "normals": normals.astype(p_type),
    }
    return out_dict


def sample_iou_points_and_sdf_vals(
    mesh, n_iou_points, cube_dim=32, padding=0, p_type=np.float16
):
    points = (np.random.rand(n_iou_points, 3).astype(np.float32) - 0.5) * (
        cube_dim + padding
    )

    sdf_vals = mesh_to_sdf.mesh_to_sdf(mesh, query_points=points)

    occ = (sdf_vals < 0).astype(int)
    points = points.astype(p_type)

    out_dict = {
        "points": points,
        "occupancies": np.packbits(occ),
    }

    return out_dict, sdf_vals


parent_out_dir = "..."
path = ""
out_dir = "./"
df = pd.read_csv(path)

for i, row in df.iterrows():
    in_mesh_path = row["mesh_path_noalign"]
    cellid = row["CellId"]
    # out_dir = f"{parent_out_dir}/{cellid}"

    mesh = trimesh.load(in_mesh_path)
    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1]) / 2
    scale_factor = (bbox[1] - bbox[0]).max()

    prep_mesh = mesh.apply_translation(-loc)
    prep_mesh = prep_mesh.apply_scale(1 / scale_factor)

    surface_data_dict = sample_points(prep_mesh, int(32**3))
    surface_data_dict["loc"] = loc
    surface_data_dict["scale"] = scale_factor

    volume_data_dict, sdf_vals = sample_iou_points_and_sdf_vals(
        prep_mesh, int(32**3), cube_dim=1
    )
    volume_data_dict["loc"] = loc
    volume_data_dict["scale"] = scale_factor

    np.savez(f"{out_dir}/points", **volume_data_dict)
    np.savez(f"{out_dir}/pointcloud", **surface_data_dict)
    np.save(f"{out_dir}/df.npy", sdf_vals)
