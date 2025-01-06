import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from aicsimageio import AICSImage
from monai.transforms import FillHoles
from tqdm import tqdm

from br.data.utils import get_mesh_from_image


def get_bounds(r):
    return_df = {"x_delta": [], "y_delta": [], "z_delta": [], "cell_id": [], "max_delta": []}
    cellid = r["CellId"]

    hole_fill_transform = FillHoles()
    seg = AICSImage(r["crop_seg_masked"]).data.squeeze()
    seg = hole_fill_transform(seg).numpy()

    mesh, _, _ = get_mesh_from_image(seg, sigma=0, lcc=False, denoise=False)

    bounds = mesh.GetBounds()
    x_delta = bounds[1] - bounds[0]
    y_delta = bounds[3] - bounds[2]
    z_delta = bounds[5] - bounds[4]
    max_delta = max([x_delta, y_delta, z_delta])
    return_df["x_delta"].append(x_delta)
    return_df["y_delta"].append(y_delta)
    return_df["z_delta"].append(z_delta)
    return_df["cell_id"].append(cellid)
    return_df["max_delta"].append(max_delta)

    return return_df


def main(args):
    # make save path directory
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)

    if args.global_path:
        df["crop_seg_masked"] = df["crop_seg_masked"].apply(lambda x: args.global_path + x)

    all_rows = []
    for _, row in df.iterrows():
        all_rows.append(row)

    with Pool(1) as p:
        jobs = tuple(
            tqdm(
                p.imap_unordered(
                    get_bounds,
                    all_rows,
                ),
                total=len(all_rows),
                desc="get_bounds",
            )
        )
        jobs = [i for i in jobs if i is not None]
        return_df = pd.DataFrame(jobs).reset_index(drop=True)

    return_df.to_csv(Path(args.save_path) / Path("bounds.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for getting global mesh bounds for polymorphic structures from WTC-11 hIPS single cell image dataset"
    )
    parser.add_argument("--save_path", type=str, required=True, help="Path to save results.")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to preprocessed single cell image manifest.",
    )
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
    python get_max_bounding_box.py --save_path './test_img/' --manifest ""../../../../../morphology_appropriate_representation_learning/preprocessed_data/npm1/manifest.csv" --global_path "../../../../../"
    """
