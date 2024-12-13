import argparse
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from pyntcloud import PyntCloud
from scipy.ndimage import binary_dilation
from tqdm import tqdm

warnings.filterwarnings("ignore")


SKEW_EXP_DICT = {
    "endosomes": 500,
    "peroxisomes": 500,
    "centrioles": 500,
    "RAB5A": 500,
    "SLC25A17": 500,
    "CETN2": 500,
}
REP_DICT = {
    "endosomes": True,
    "peroxisomes": True,
    "centrioles": True,
    "RAB5A": True,
    "SLC25A17": True,
    "CETN2": True,
}


def compute_labels(row, save=True):
    num_points = 20480
    path = row["crop_raw"]
    structure_name = row["structure_name"]
    img_full = AICSImage(path).data[0]
    raw = img_full[2]  # raw struct

    path = row["crop_seg"]
    img_seg = AICSImage(path).data[0]
    mem = img_seg[1]

    dilation_shape = (8, 8, 8)
    binary_structure = np.ones(dilation_shape)
    mem = binary_dilation(mem, structure=binary_structure)

    center = get_center_of_mass(mem)
    z_center, y_center, x_center = center[0], center[1], center[2]
    raw = np.where(mem, raw, raw.min())

    raw = np.where(raw <= np.unique(raw).mean(), 0, raw)

    z, y, x = np.where(np.ones_like(raw) > 0)
    probs = raw.copy()
    probs_orig = probs.copy()
    probs_orig = probs_orig.flatten()
    probs = probs.flatten()
    probs = probs / probs.max()

    # sampling based on raw images
    skewness = (
        SKEW_EXP_DICT[structure_name] * (3 * (probs.mean() - np.median(probs))) / probs.std()
    )

    if skewness < 25:
        skewness = 25
    probs = np.exp(skewness * probs)

    # set prob to 0 outside nuclear mask
    inds = np.where(mem.flatten() == 0)[0]
    probs[inds] = 0

    # scalr prob so it sums to 1
    probs = probs / probs.sum()

    replace = REP_DICT[structure_name]

    idxs = np.random.choice(np.arange(len(probs)), size=num_points, replace=replace, p=probs)
    # noise important to avoid nans during encoding
    disp = 0.001
    x = x[idxs] + (np.random.rand(len(idxs)) - 0.5) * disp
    y = y[idxs] + (np.random.rand(len(idxs)) - 0.5) * disp
    z = z[idxs] + (np.random.rand(len(idxs)) - 0.5) * disp

    probs = probs[idxs]
    probs_orig = probs_orig[idxs]
    new_cents = np.stack([z, y, x, probs], axis=1)
    new_cents = pd.DataFrame(new_cents, columns=["z", "y", "x", "s"])
    assert new_cents.shape[0] == num_points
    new_cents["s"] = probs_orig
    if not save:
        return new_cents, raw, np.concatenate([img_full, img_seg], axis=0), center
    new_cents["z"] = new_cents["z"] - z_center
    new_cents["y"] = new_cents["y"] - y_center
    new_cents["x"] = new_cents["x"] - x_center

    cell_id = str(row["CellId"])

    save_path = Path(path_prefix) / Path(cell_id + ".ply")

    new_cents = new_cents.astype(float)
    cloud = PyntCloud(new_cents)
    cloud.to_file(str(save_path))


def get_center_of_mass(img):
    center_of_mass = np.mean(np.stack(np.where(img > 0)), axis=1)
    return np.floor(center_of_mass + 0.5).astype(int)


def main(args):

    # make save path directory
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.preprocessed_manifest)
    df = df.loc[df["structure_name"].isin(SKEW_EXP_DICT.keys())]

    if args.global_path:
        df["registered_path"] = df["registered_path"].apply(lambda x: args.global_path + x)

    global path_prefix
    path_prefix = args.save_path

    all_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        all_rows.append(row)

    with Pool(40) as p:
        _ = tuple(
            tqdm(
                p.imap_unordered(
                    compute_labels,
                    all_rows,
                ),
                total=len(all_rows),
                desc="compute_everything",
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for computing point clouds for cytoplasmic structures from WTC-11 hIPS single cell image dataset"
    )
    parser.add_argument("--save_path", type=str, required=True, help="Path to save results.")
    parser.add_argument(
        "--global_path",
        type=str,
        default=None,
        required=False,
        help="Path to append to relative paths in preprocessed manifest",
    )
    parser.add_argument(
        "--preprocessed_manifest",
        type=str,
        required=True,
        help="Path to processed single cell image manifest.",
    )
    args = parser.parse_args()
    main(args)

    """
    Example run:

    python punctate_cyto.py --save_path "./make_pcs_test" --preprocessed_manifest "./subpackages/image_preprocessing/tmp_output_variance/processed/manifest.parquet" --global_path "./subpackages/image_preprocessing/"
    """
