import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from aicsimageio import AICSImage
from monai.transforms import FillHoles
from tqdm import tqdm

from br.analysis.analysis_utils import str2bool
from br.data.utils import (
    get_mesh_from_image,
    get_scaled_mesh,
    get_sdf_from_mesh_vtk,
    voxelize_scaled_mesh,
)


def process(r):
    cellid = r["CellId"]
    scale_factor = r["scale_factor"]
    vox_resolution = r["vox_resolution"]

    out_path_sdf = f"{out_dir_scaled_sdf}/{cellid}"
    out_path_seg = f"{out_dir_scaled_seg}/{cellid}"
    out_path_mesh = f"{out_dir_mesh}/{cellid}.stl"
    hole_fill_transform = FillHoles()
    seg = AICSImage(r["crop_seg_masked"]).data.squeeze()
    seg = hole_fill_transform(seg).numpy()

    mesh, _, _ = get_mesh_from_image(seg, sigma=0, lcc=False, denoise=False)

    pv.wrap(mesh).save(out_path_mesh)

    sdf, scale_factor = get_sdf_from_mesh_vtk(
        None, vox_resolution=vox_resolution, scale_factor=scale_factor, vpolydata=mesh
    )
    np.save(out_path_sdf, sdf)
    vox_shape = (vox_resolution, vox_resolution, vox_resolution)
    scaled_mesh, _ = get_scaled_mesh(None, int(vox_resolution), scale_factor, mesh, True)

    scaled_seg = voxelize_scaled_mesh(scaled_mesh)
    com = scaled_seg.shape
    pad = []
    for i, j in zip(vox_shape, com):
        pad.append((int(i - j) // 2, int(i - j) // 2))
    scaled_seg = np.pad(scaled_seg, pad)

    np.save(out_path_seg, scaled_seg)


def main(args):

    # make save path directory
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    global out_dir_scaled_seg
    global out_dir_scaled_sdf
    global out_dir_mesh

    out_dir_scaled_seg = Path(args.save_path) / Path("outputs_seg")
    out_dir_scaled_sdf = Path(args.save_path) / Path("outputs_sdf")
    out_dir_mesh = Path(args.save_path) / Path("outputs_mesh")

    Path(out_dir_scaled_seg).mkdir(parents=True, exist_ok=True)
    Path(out_dir_scaled_sdf).mkdir(parents=True, exist_ok=True)
    Path(out_dir_mesh).mkdir(parents=True, exist_ok=True)

    scale_factor_path = Path(args.manifest).parent / Path("scale_factor.npz")

    sc_factor_data = np.load(scale_factor_path, allow_pickle=True)
    scale_factor_dict = dict(zip(sc_factor_data["keys"], sc_factor_data["values"]))

    df = pd.read_csv(args.manifest)

    if args.global_path:
        df["crop_seg_masked"] = df["crop_seg_masked"].apply(lambda x: args.global_path + x)

    if args.debug:
        df = df.sample(n=5).reset_index(drop=True)

    all_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        row["scale_factor"] = scale_factor_dict[row["CellId"]]
        row["vox_resolution"] = args.vox_resolution
        all_rows.append(row)

    with Pool(40) as p:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for computing scaled segmentations and SDFs for polymorphic structures from WTC-11 hIPS single cell image dataset"
    )
    parser.add_argument("--save_path", type=str, required=True, help="Path to save results.")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to preprocessed single cell image manifest.",
    )
    parser.add_argument(
        "--vox_resolution",
        type=int,
        required=True,
        help="Resolution to voxelize images to",
    )
    parser.add_argument("--debug", type=str2bool, default=False, help="Enable debug mode.")
    parser.add_argument(
        "--global_path",
        type=str,
        default=None,
        required=False,
        help="Path to append to relative paths in preprocessed manifest",
    )

    args = parser.parse_args()
    main(args)

    """
    Example run:

    python image_sdfs.py --save_path "./test_img/" --manifest "/allen/aics/modeling/ritvik/projects/latest_clones/benchmarking_representations/morphology_appropriate_representation_learning/preprocessed_data/npm1/manifest.csv" --vox_resolution 32 --debug True --global_path "../../../../../"
    """
