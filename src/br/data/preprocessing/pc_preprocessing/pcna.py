from multiprocessing import Pool
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from scipy.ndimage import binary_dilation
from skimage.io import imread
from tqdm import tqdm


def compute_labels(row, save=True):
    path = row["registered_path"]

    num_points = 20480
    img = imread(path)

    img_nuc = img[6]  # nuc seg channel
    raw = img[2]  # PCNA intensity channel

    center = get_center_of_mass(img_nuc)
    z_center, y_center, x_center = center[0], center[1], center[2]
    raw = np.where(raw < 60000, raw, raw.min())

    dilation_shape = (8, 8, 8)
    binary_structure = np.ones(dilation_shape)
    img_nuc = binary_dilation(img_nuc, structure=binary_structure)
    raw = np.where(img_nuc, raw, raw.min())

    z, y, x = np.where(np.ones_like(raw) > 0)
    probs = raw.copy()
    probs_orig = probs.copy()
    probs_orig = probs_orig.flatten()
    probs = probs.flatten()
    probs = probs / probs.max()

    skewness = 100 * (3 * (probs.mean() - np.median(probs))) / probs.std()
    probs = np.exp(skewness * probs)

    # set prob to 0 outside nuclear mask
    inds = np.where(img_nuc.flatten() == 0)[0]
    probs[inds] = 0

    # scalr prob so it sums to 1
    probs = probs / probs.sum()

    idxs = np.random.choice(np.arange(len(probs)), size=num_points, replace=False, p=probs)
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
        return new_cents, raw, img, center

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

    if args.global_path:
        df["registered_path"] = df["registered_path"].apply(
            lambda x: args.global_path + x
        )

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
    parser = argparse.ArgumentParser(description="Script for computing point clouds for PCNA dataset")
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
    python pcna.py --save_path "./make_pcs_test" --preprocessed_manifest "/allen/aics/modeling/ritvik/projects/latest_clones/benchmarking_representations/subpackages/image_preprocessing/tmp_output_pcna/processed/manifest.parquet" --global_path "/allen/aics/modeling/ritvik/projects/latest_clones/benchmarking_representations/subpackages/image_preprocessing/
    """