import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import mesh_to_sdf
import numpy as np
import trimesh
from tqdm import tqdm

os.environ["PYOPENGL_PLATFORM"] = "egl"


def sample_points(mesh, n_points, p_type=np.float16):
    pcl, idx = mesh.sample(n_points, return_index=True)
    normals = mesh.face_normals[idx]

    out_dict = {
        "points": pcl.astype(p_type),
        "normals": normals.astype(p_type),
    }
    return out_dict


def sample_iou_points_and_sdf_vals(mesh, n_iou_points, cube_dim=32, padding=0, p_type=np.float16):
    points = (np.random.rand(n_iou_points, 3).astype(np.float32) - 0.5) * (cube_dim + padding)

    sdf_vals = mesh_to_sdf.mesh_to_sdf(mesh, query_points=points)

    occ = (sdf_vals < 0).astype(int)
    points = points.astype(p_type)

    out_dict = {
        "points": points,
        "occupancies": np.packbits(occ),
    }

    return out_dict, sdf_vals


def process_mesh(cellid):
    in_mesh_path = Path(mesh_path) / Path(cellid + ".stl")

    this_out_dir = out_dir / Path(cellid)
    out_dir_points = Path(this_out_dir) / Path("points")
    out_dir_pointcloud = Path(this_out_dir) / Path("pointcloud")

    Path(this_out_dir).mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(in_mesh_path)
    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1]) / 2
    scale_factor = (bbox[1] - bbox[0]).max()

    prep_mesh = mesh.apply_translation(-loc)
    prep_mesh = prep_mesh.apply_scale(1 / scale_factor)

    surface_data_dict = sample_points(prep_mesh, int(vox_resolution**3))
    surface_data_dict["loc"] = loc
    surface_data_dict["scale"] = scale_factor

    volume_data_dict, sdf_vals = sample_iou_points_and_sdf_vals(
        prep_mesh, int(vox_resolution**3), cube_dim=1
    )
    volume_data_dict["loc"] = loc
    volume_data_dict["scale"] = scale_factor

    np.savez(out_dir_points, **volume_data_dict)
    np.savez(out_dir_pointcloud, **surface_data_dict)
    np.save(f"{this_out_dir}/df.npy", sdf_vals)


def main(args):
    # make save path directory
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    global out_dir
    global vox_resolution
    global mesh_path

    mesh_path = args.scaled_mesh_path

    vox_resolution = args.vox_resolution

    out_dir = Path(args.save_path)

    cell_ids_to_process = [i.split(".")[0] for i in os.listdir(mesh_path)]

    with Pool(1) as p:
        _ = tuple(
            tqdm(
                p.imap_unordered(
                    process_mesh,
                    cell_ids_to_process,
                ),
                total=len(cell_ids_to_process),
                desc="compute_everything",
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for computing SDF pointclouds for polymorphic structures from WTC-11 hIPS single cell image dataset"
    )
    parser.add_argument("--save_path", type=str, required=True, help="Path to save results.")
    parser.add_argument(
        "--scaled_mesh_path",
        type=str,
        required=True,
        help="Path to folder containing scaled meshes",
    )
    parser.add_argument(
        "--vox_resolution",
        type=int,
        required=True,
        help="Resolution to voxelize images to",
    )

    args = parser.parse_args()
    main(args)

    """
    Example run:
    python pc_sdfs.py --save_path "./test_pcs/" --scaled_mesh_path "./test_img/outputs_mesh/" --vox_resolution 32
    """
