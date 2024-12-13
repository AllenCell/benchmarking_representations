import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from br.analysis.analysis_utils import plot_image, plot_pointcloud
from br.data.preprocessing.pc_preprocessing.pcna import (
    compute_labels as compute_labels_pcna,
)
from br.data.preprocessing.pc_preprocessing.punctate_cyto import (
    compute_labels as compute_labels_var_cyto,
)
from br.data.preprocessing.pc_preprocessing.punctate_nuc import (
    compute_labels as compute_labels_var_nuc,
)
from br.features.utils import normalize_intensities_and_get_colormap_apply

dataset_dict = {
    "pcna": {"raw_ind": 2, "nuc_ind": 6, "mem_ind": None},
    "other_punctate": {"raw_ind": 2, "nuc_ind": 3, "mem_ind": 4},
}

viz_norms = {
    "CETN2": [440, 800],
    "NUP153": [420, 600],
    "HIST1H2BJ": [450, 2885],
    "SON": [420, 1500],
    "SLC25A17": [400, 515],
    "RAB5A": [420, 600],
    "SMC1A": [450, 630],
}

cell_ids_ = {
    "pcna": [
        "7624cd5b-715a-478e-9648-3bac4a73abe8",
        "80d40c5e-65bf-43b0-8dea-b697c421ea78",
        "6a3ab51f-fa68-4fe1-a13b-2b2461ed71b4",
        "aabbbca4-6c35-4f3d-9467-7d573482f236",
        "d23de56e-bacf-4ec8-8e18-39822fea777b",
        "c382794f-5baf-4b17-8574-62dccbbbaefc",
        "50b52c3e-4756-4684-a281-0141525ded9f",
        "8713eea5-da72-4644-96fe-ba8340edb67d",
    ],
    "other_punctate": [721646, 873680, 994027, 490385, 451974, 811336, 835431],
}


def main(args):

    # make save path directory
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    orig_image_df = pd.read_parquet(args.preprocessed_manifest)

    if args.global_path:
        orig_image_df["registered_path"] = orig_image_df["registered_path"].apply(
            lambda x: args.global_path + x
        )

    assert args.dataset_name in ["pcna", "other_punctate"]
    dataset_ = dataset_dict[args.dataset_name]
    raw_ind = dataset_["raw_ind"]
    nuc_ind = dataset_["nuc_ind"]
    mem_ind = dataset_["mem_ind"]

    strat = args.class_column
    strat_val = args.class_label

    if not strat:
        cell_ids = cell_ids_[args.dataset_name]
        orig_image_df = orig_image_df.loc[orig_image_df["CellId"].isin(cell_ids)].reset_index(
            drop=True
        )
    else:
        orig_image_df = orig_image_df.loc[orig_image_df[strat] == strat_val].sample(n=1)

    for _, this_image in orig_image_df.iterrows():
        cell_id = this_image["CellId"]
        if not strat:
            strat_val = this_image["structure_name"]

        if args.dataset_name == "pcna":
            points_all, _, img, center = compute_labels_pcna(this_image, False)
            vmin, vmax = None, None
            num_slices = 15
            center_slice = True
        elif args.dataset_name == "other_punctate":
            assert strat == "structure_name"
            if strat_val in ["CETN2", "RAB5A", "SLC25A17"]:
                points_all, _, img, center = compute_labels_var_cyto(this_image, False)
                center_slice = False
                num_slices = None
            else:
                center_slice = True
                points_all, _, img, center = compute_labels_var_nuc(this_image, False)
                num_slices = 1
            this_viz_norm = viz_norms[strat_val]
            vmin = this_viz_norm[0]
            vmax = this_viz_norm[1]

        img_raw = img[raw_ind]
        img_nuc = img[nuc_ind]
        img_raw = np.where(img_raw < 60000, img_raw, img_raw.min())
        img_mem = img_nuc
        if mem_ind is not None:
            img_mem = img[mem_ind]

        if (args.dataset_name == "other_punctate") and (
            strat_val in ["CETN2", "RAB5A", "SLC25A17"]
        ):
            img_raw = np.where(img_mem, img_raw, 0)  # mask by mem/nuc seg
        else:
            img_raw = np.where(img_nuc, img_raw, 0)  # mask by mem/nuc seg

        # Sample sparse point cloud and get images
        probs2 = points_all["s"].values
        probs2 = np.where(probs2 < 0, 0, probs2)
        probs2 = probs2 / probs2.sum()
        idxs2 = np.random.choice(np.arange(len(probs2)), size=2048, replace=True, p=probs2)
        points = points_all.iloc[idxs2].reset_index(drop=True)

        if not vmin:
            vmin = points["s"].min()
            vmax = points["s"].max()
        points = normalize_intensities_and_get_colormap_apply(points, vmin, vmax)
        points_all = normalize_intensities_and_get_colormap_apply(points_all, vmin, vmax)

        save = True
        fig, ax_array = plt.subplots(1, 3, figsize=(10, 5))

        ax_array, z_interp = plot_image(
            ax_array,
            img_raw,
            img_nuc,
            img_mem,
            vmin,
            vmax,
            num_slices=num_slices,
            show_nuc_countour=True,
        )
        ax_array[0].set_title("Raw image")

        name = strat_val + "_" + str(cell_id)

        plot_pointcloud(
            ax_array[1],
            points_all,
            z_interp,
            plt.get_cmap("YlGnBu"),
            save_path=args.save_path,
            name=name,
            center=center,
            save=False,
            center_slice=center_slice,
        )
        ax_array[1].set_title("Sampling dense PC")

        plot_pointcloud(
            ax_array[2],
            points,
            z_interp,
            plt.get_cmap("YlGnBu"),
            save_path=args.save_path,
            name=name,
            center=center,
            save=save,
            center_slice=center_slice,
        )
        ax_array[2].set_title("Sampling sparse PC")
        print(f"Saving {name}.png")
        fig.savefig(Path(args.save_path) / Path(f"{name}.png"), bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for computing features")
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
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument(
        "--class_column",
        type=str,
        default=None,
        required=False,
        help="Column name of class to use for sampling, e.g. cell_stage_fine",
    )
    parser.add_argument(
        "--class_label",
        type=str,
        default=None,
        required=False,
        help="Specific class label to sample, e.g. lateS",
    )
    args = parser.parse_args()

    # Validate that required paths are provided
    if not args.save_path or not args.dataset_name:
        print("Error: Required arguments are missing.")
        sys.exit(1)

    main(args)

    """
    Example run:

    PCNA dataset
    python visualize_pointclouds.py --save_path "./plot_pcs_test" --preprocessed_manifest "./subpackages/image_preprocessing/tmp_output_pcna/processed/manifest.parquet" --dataset_name "pcna" --class_column "cell_stage_fine" --class_label "lateS" --global_path "./subpackages/image_preprocessing/"

    Other punctate dataset
    python visualize_pointclouds.py --save_path "./plot_pcs_test" --preprocessed_manifest "./subpackages/image_preprocessing/tmp_output_variance/processed/manifest.parquet" --dataset_name "other_punctate" --class_column "structure_name" --class_label "CETN2" --global_path "./subpackages/image_preprocessing/"
    """
