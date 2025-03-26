import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from br.models.compute_features import get_embeddings
from br.models.utils import get_all_configs_per_dataset
from skimage import measure as skmeasure
from skimage import morphology as skmorpho
from tqdm import tqdm
from br.features.classification import get_classification_df
import matplotlib.pyplot as plt


def get_surface_area(input_img):
    # Forces a 1 pixel-wide offset to avoid problems with binary
    # erosion algorithm
    input_img[:, :, [0, -1]] = 0
    input_img[:, [0, -1], :] = 0
    input_img[[0, -1], :, :] = 0
    input_img_surface = np.logical_xor(
        input_img, skmorpho.binary_erosion(input_img)
    ).astype(np.uint8)
    # Loop through the boundary voxels to calculate the number of
    # boundary faces. Using 6-neighborhod.
    pxl_z, pxl_y, pxl_x = np.nonzero(input_img_surface)
    dx = np.array([0, -1, 0, 1, 0, 0])
    dy = np.array([0, 0, 1, 0, -1, 0])
    dz = np.array([-1, 0, 0, 0, 0, 1])
    surface_area = 0
    for (k, j, i) in zip(pxl_z, pxl_y, pxl_x):
        surface_area += 6 - (input_img[k + dz, j + dy, i + dx] > 0).sum()
    return int(surface_area)


def get_basic_features(img):
    features = {}
    input_image = img.copy()
    input_image = (input_image > 0).astype(np.uint8)
    input_image_lcc = skmeasure.label(input_image)
    features["connectivity_cc"] = input_image_lcc.max()
    if features["connectivity_cc"] > 0:
        counts = np.bincount(input_image_lcc.reshape(-1))
        lcc = 1 + np.argmax(counts[1:])
        input_image_lcc[input_image_lcc != lcc] = 0
        input_image_lcc[input_image_lcc == lcc] = 1
        input_image_lcc = input_image_lcc.astype(np.uint8)
        for img, suffix in zip([input_image, input_image_lcc], ["", "_lcc"]):
            z, y, x = np.where(img)
            features[f"shape_volume{suffix}"] = img.sum()
            features[f"position_depth{suffix}"] = 1 + np.ptp(z)
            features[f"position_height{suffix}"] = 1 + np.ptp(y)
            features[f"position_width{suffix}"] = 1 + np.ptp(x)
            for uname, u in zip(["x", "y", "z"], [x, y, z]):
                features[f"position_{uname}_centroid{suffix}"] = u.mean()
            features[f"roundness_surface_area{suffix}"] = get_surface_area(img)
    else:
        for img, suffix in zip([input_image, input_image_lcc], ["", "_lcc"]):
            features[f"shape_volume{suffix}"] = np.nan
            features[f"position_depth{suffix}"] = np.nan
            features[f"position_height{suffix}"] = np.nan
            features[f"position_width{suffix}"] = np.nan
            for uname in ["x", "y", "z"]:
                features[f"position_{uname}_centroid{suffix}"] = np.nan
            features[f"roundness_surface_area{suffix}"] = np.nan
    return features


def main(args):

    config_path = os.environ.get("CYTODL_CONFIG_PATH")
    results_path = config_path + "/results/"
    DATASET_INFO = get_all_configs_per_dataset(results_path)

    all_ret, orig_df = get_embeddings(
        ["Rotation_invariant_pointcloud_SDF"],
        args.dataset_name,
        DATASET_INFO,
        args.embeddings_path,
    )
    orig_df["volume_of_nucleus_um3"] = orig_df["dna_shape_volume_lcc"] * 0.108**3

    bins = [
        (247.407, 390.752),
        (390.752, 533.383),
        (533.383, 676.015),
        (676.015, 818.646),
        (818.646, 961.277),
    ]
    correct_bins = []
    for ind, row in orig_df.iterrows():
        this_bin = []
        for bin_ in bins:
            if (row["volume_of_nucleus_um3"] > bin_[0]) and (
                row["volume_of_nucleus_um3"] <= bin_[1]
            ):
                this_bin.append(bin_)
        if row["volume_of_nucleus_um3"] < bins[0][0]:
            this_bin.append(bin_)
        if row["volume_of_nucleus_um3"] > bins[4][1]:
            this_bin.append(bin_)
        assert len(this_bin) == 1
        correct_bins.append(this_bin[0])
    orig_df["vol_bins"] = correct_bins
    orig_df["vol_bins_inds"] = pd.factorize(orig_df["vol_bins"])[0]

    all_feats = []
    for ind, row in tqdm(orig_df.iterrows(), total=len(orig_df)):
        img = np.load(row["seg_path"])
        feats = get_basic_features(img)
        feats["CellId"] = row["CellId"]
        all_feats.append(pd.DataFrame(feats, index=[0]))
    all_feats = pd.concat(all_feats, axis=0)
    all_feats = all_feats.merge(
        orig_df[["CellId", "vol_bins", "vol_bins_inds"]], on="CellId"
    )
    all_feats["mean_volume"] = all_feats["shape_volume"] / all_feats["connectivity_cc"]
    all_feats["mean_surface_area"] = (
        all_feats["roundness_surface_area"] / all_feats["connectivity_cc"]
    )

    all_feats = all_feats.merge(
        orig_df[["CellId", "STR_connectivity_cc_thresh"]], on="CellId"
    )
    all_feats = all_feats.loc[all_feats["CellId"] != 724520].reset_index(
        drop=True
    )  # nan row
    all_ret = all_ret.loc[all_ret["CellId"] != 724520].reset_index(drop=True)  # nan row
    assert not all_feats["mean_surface_area"].isna().any()

    all_ret = all_ret.merge(
        orig_df[["CellId", "vol_bins", "vol_bins_inds"]],
        on="CellId",
    )
    from br.features.classification import get_classification_df

    all_baseline = []
    all_feats["model"] = "baseline"
    for bin in all_feats["vol_bins"].unique():
        this = all_feats.loc[all_feats["vol_bins"] == bin].reset_index(drop=True)
        baseline = get_classification_df(
            this,
            "STR_connectivity_cc_thresh",
            None,
            ["mean_volume", "mean_surface_area"],
        )
        baseline["vol_bin"] = str(bin)
        all_baseline.append(baseline)
    all_baseline = pd.concat(all_baseline, axis=0)

    all_ret["model"] = "reps"
    all_reps = []
    for bin in all_ret["vol_bins"].unique():
        this = all_ret.loc[all_ret["vol_bins"] == bin].reset_index(drop=True)
        reps = get_classification_df(this, "STR_connectivity_cc_thresh", None, None)
        reps["vol_bin"] = str(bin)
        all_reps.append(reps)
    all_reps = pd.concat(all_reps, axis=0)
    all_reps["features"] = "Rotation invariant point cloud representation"
    all_baseline["features"] = "Mean nucleoli volume and surface area"
    plot = pd.concat([all_reps, all_baseline], axis=0)
    map_ = {
        "reps": "Rotation invariant point cloud representation",
        "baseline": "Mean nucleoli volume and surface area",
    }
    plot["model"] = plot["model"].replace(map_)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    x_order = [
        "(247.407, 390.752)",
        "(390.752, 533.383)",
        "(533.383, 676.015)",
        "(676.015, 818.646)",
        "(818.646, 961.277)",
    ]
    g = sns.boxplot(
        ax=ax, data=plot, x="vol_bin", y="top_1_acc", hue="features", order=x_order
    )
    plt.xticks(rotation=30)
    ax.set_xticklabels(
        ["0", "1", "2", "3", "4"], rotation=0
    )  # Set tick labels, rotate for readability
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Volume bin")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.savefig(
        args.save_path + "classification_number_pieces.png",
        bbox_inches="tight",
        dpi=300,
    )
    # fig.savefig("classification_number_pieces_nogrouping.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for computing perturbation detection metrics"
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the results."
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to the saved embeddings.",
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset."
    )
    args = parser.parse_args()

    # Validate that required paths are provided
    if not args.save_path or not args.embeddings_path:
        print("Error: Required arguments are missing.")
        sys.exit(1)

    main(args)

    """
    Example runs for each dataset:

    cellpack dataset
    python src/br/analysis/run_classification.py --save_path "./outputs_npm1/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/npm1/" --dataset_name "npm1"
    """
