import argparse
import os
import sys
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import pandas as pd
from br.models.compute_features import get_embeddings
from br.models.utils import get_all_configs_per_dataset
from skimage import measure
import seaborn as sns


def pad_to_size(array, target_shape, padding_value=0):
    """Pads a NumPy array to a specific size.

    Args:
        array: The NumPy array to pad.
        target_shape: The desired shape of the padded array (tuple).
        padding_value: The value to use for padding (default is 0).

    Returns:
        A new NumPy array with the specified target shape, padded as necessary.
    """
    current_shape = np.array(array.shape)
    target_shape = np.array(target_shape)

    if np.all(current_shape >= target_shape):
        return array  # No padding needed

    padding_needed = np.maximum(0, target_shape - current_shape)
    padding_before = padding_needed // 2
    padding_after = padding_needed - padding_before

    padding = [(padding_before[i], padding_after[i]) for i in range(len(target_shape))]

    padded_array = np.pad(
        array, pad_width=padding, mode="constant", constant_values=padding_value
    )
    return padded_array


def get_image(cell_id, raw_df):
    this = raw_df.loc[raw_df["CellId"] == cell_id].reset_index(drop=True)
    img_raw = imread(this["crop_raw"].iloc[0]).max(0)
    img_nuc_seg = imread(this["crop_seg_nuc"].iloc[0]).max(0)
    return img_raw, img_nuc_seg


def sort_by_second_element_with_index(data):
    """
    Sorts a list of lists based on the second element of each sublist
    and returns a list of original indices in the sorted order.

    Args:
        data: A list of lists.

    Returns:
        A list of integers representing the original indices
        of the sublists after sorting.
    """
    indexed_data = list(enumerate(data))
    indexed_data.sort(key=lambda x: x[1][1])
    sorted_indices = [index for index, _ in indexed_data]
    return sorted_indices


def merge_contours(contours, distance_threshold):
    merged_contours = list(contours)  # Create a copy to modify
    while True:
        merged_count = 0
        new_contours = []
        used_indices = set()

        for i in range(len(merged_contours)):
            if i in used_indices:
                continue

            contour1 = np.array(merged_contours[i])
            closest_contour_index = -1
            min_distance = float("inf")

            for j in range(i + 1, len(merged_contours)):
                if j in used_indices:
                    continue
                contour2 = np.array(merged_contours[j])

                # Calculate distances between all pairs of points
                distances = np.sqrt(
                    (
                        (contour1[:, 0, None] - contour2[:, 0]) ** 2
                        + (contour1[:, 1, None] - contour2[:, 1]) ** 2
                    )
                )

                # Find the minimum distance
                distance = np.min(distances)

                if distance < min_distance:
                    min_distance = distance
                    closest_contour_index = j

            if closest_contour_index != -1 and min_distance < distance_threshold:
                # Merge contours
                merged_contour = np.concatenate(
                    (contour1, np.array(merged_contours[closest_contour_index]))
                )
                new_contours.append(merged_contour)
                used_indices.add(closest_contour_index)
                used_indices.add(i)
                merged_count += 1
            else:
                new_contours.append(contour1)
        merged_contours = new_contours
        if merged_count == 0:
            break
    return merged_contours


def main(args):

    save_path = args.save_path
    config_path = os.environ.get("CYTODL_CONFIG_PATH")
    results_path = config_path + "/results/"
    DATASET_INFO = get_all_configs_per_dataset(results_path)

    all_ret, _ = get_embeddings(
        ["Rotation_invariant_pointcloud_SDF"],
        args.dataset_name,
        DATASET_INFO,
        args.embeddings_path,
    )
    raw_df = pd.read_csv(Path(args.raw_path) / "manifest.csv")
    raw_df["crop_raw"] = raw_df["crop_raw"].apply(
        lambda x: Path(args.raw_path) / Path(x)
    )
    raw_df["crop_seg_nuc"] = raw_df["crop_seg_nuc"].apply(
        lambda x: Path(args.raw_path) / Path(x)
    )

    map_ = {
        "Actinomyocin D 0.5ug per mL": "Actinomyocin D",
        "Jasplakinolide 50 nM (E5)": "Jasplakinolide",
        "Paclitaxel 5uM (E2)": "Paclitaxel",
        "Staurosporine 1uM (E8)": "Staurosporine",
        "Nocodazole 0.1uM (E4)": "Nocodazole",
        "Roscovitine 10uM (E9)": "Roscovitine 10uM",
        "Torin 1uM": "Torin",
        "Rapamycin 1uM (E7)": "Rapamycin",
        "H89 10uM (E3)": "H89",
        "Monensin 1.1uM": "Monensin",
        "Rotenone 0.5uM (E6)": "Rotenone",
        "Roscovitine 5uM (E10)": "Roscovitine 5uM",
        "BIX 1uM": "BIX",
        "Bafilomycin A1 0.1uM": "Bafilomycin A1",
        "Latrunculin A1 0.1uM": "Latrunculin A1",
        "Chloroquin 40uM": "Chloroquin",
        "Brefeldin 5uM": "Brefeldin",
    }
    all_ret["condition"] = all_ret["condition"].replace(map_)
    cols = [i for i in all_ret.columns if "mu" in i]

    hits = [
        "Actinomyocin D",
        "Staurosporine",
        "Paclitaxel",
        "Nocodazole",
        "Torin",
        "Jasplakinolide",
        "Roscovitine 10uM",
    ]

    res = {}
    scale_lows = [0.3, 0.3, 0.3, 0.3, 0.4, 0.25, 0.3, 0.3, 0.3, 0.3]
    scale_highs = [0.3, 0.3, 0.3, 0.3, 0.4, 0.25, 0.3, 0.3, 0.3, 0.3]
    scale_lows = [i * 0.1 for i in scale_lows]
    scale_highs = [i * 0.1 for i in scale_highs]

    n_pcs = 20
    pca = PCA(n_components=n_pcs)
    pcs = pca.fit_transform(all_ret[cols].values)
    cols = [f"pc_{i}" for i in range(n_pcs)]
    all_ret[cols] = pcs

    merge_thresh = [17] * len(hits)
    merge_thresh[2] = 10
    merge_thresh[3] = 7
    merge_thresh[4] = 10
    merge_thresh[5] = 12
    merge_thresh[6] = 7
    sns.set_context("talk")

    for j, hit in enumerate(hits):
        print("Analysis for", hit)
        scale_low = scale_lows[j]
        scale_high = scale_highs[j]
        tmp1 = all_ret.loc[all_ret["condition"] == "DMSO (control)"]
        tmp2 = all_ret.loc[all_ret["condition"] == hit]
        tmp1["class"] = 0
        tmp2["class"] = 1
        tmp = pd.concat([tmp1, tmp2], axis=0).reset_index(drop=True)
        clf = LDA()
        X = tmp[cols].values
        y = tmp["class"].values

        preds = clf.fit_transform(X, y)
        lda_direction = clf.coef_[0]
        lda_line = np.array([-lda_direction * scale_low, lda_direction * scale_high])
        res[hit] = preds

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, 8))
        # PCA Projection plot
        ax1.scatter(
            tmp1["pc_0"].values,
            tmp1["pc_1"].values,
            c=[colors[0]],
            label="control",
            alpha=0.7,
            edgecolors="none",
        )
        ax1.scatter(
            tmp2["pc_0"].values,
            tmp2["pc_1"].values,
            c=[colors[1]],
            label=hit,
            alpha=0.7,
            edgecolors="none",
        )

        arrow_start = lda_line[0]
        arrow_end = lda_line[1]
        arrow_vector = arrow_end - arrow_start

        ax1.arrow(
            arrow_start[0],
            arrow_start[1],
            arrow_vector[0],
            arrow_vector[1],
            color="r",
            width=0.01,
            head_width=0.05,
            head_length=0.05,
            length_includes_head=True,
            label="LDA direction",
        )
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        # ax1.legend(fontsize=10)
        ax1.set_title("PCA Projection", fontweight="bold")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        # ax1.set_aspect('equal')
        ax1.axis("square")

        # LDA Projection plot
        ax2.hist(
            preds[y == 0],
            bins=50,
            color=colors[0],
            density=True,
            label="control",
            alpha=0.5,
        )
        ax2.hist(
            preds[y == 1],
            bins=50,
            color=colors[1],
            density=True,
            label=hit,
            alpha=0.5,
        )
        ax2.set_xlabel("LDA Coordinates")
        ax2.set_ylabel("Density")
        # ax2.legend(fontsize=10)
        ax2.set_title("LDA Projection", fontweight="bold")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        walk = np.linspace(lda_line[0], lda_line[1], 7)
        movie = []
        count = 0
        max_x, max_y = 0, 0
        seen = set()
        for w in tqdm(walk, desc="Traversing PC-LDA line"):
            dist = np.linalg.norm(X - w, axis=1)
            dist_argsort = np.argsort(dist)
            ind = 0
            idx = dist_argsort[ind]
            this_id = tmp.iloc[idx]["CellId"]

            while this_id in seen:
                ind += 1
                idx = dist_argsort[ind]
                this_id = tmp.iloc[idx]["CellId"]

            seen.add(this_id)
            img, _ = get_image(this_id, raw_df)
            max_x = max(max_x, img.shape[0])
            max_y = max(max_y, img.shape[1])

        seen = set()
        movie2 = []
        classes = []
        from scipy import ndimage

        for w in tqdm(walk, desc="Traversing PC-LDA line"):
            dist = np.linalg.norm(X - w, axis=1)
            dist_argsort = np.argsort(dist)
            examples = []
            nuc_examples = []

            ind = 0
            idx = dist_argsort[ind]
            this_id = tmp.iloc[idx]["CellId"]

            while this_id in seen:
                ind += 1
                idx = dist_argsort[ind]
                this_id = tmp.iloc[idx]["CellId"]

            seen.add(this_id)

            this_class = tmp.iloc[idx]["class"]
            classes.append(this_class)

            ax1.scatter(
                tmp.iloc[idx]["pc_0"],
                tmp.iloc[idx]["pc_1"],
                marker="*",
                label="retrieved",
                alpha=0.7,
                edgecolors="k",
            )
            img, seg = get_image(this_id, raw_df)
            seg = ndimage.binary_fill_holes(seg).astype(int)
            img = pad_to_size(img, [max_x, max_y], padding_value=0)
            seg = pad_to_size(seg, [max_x, max_y], padding_value=0)
            examples.append(img)
            nuc_examples.append(seg)
            examples = np.vstack(examples)
            nuc_examples = np.vstack(nuc_examples)

            movie.append(examples)

            movie2.append(nuc_examples)

            count += 1
        movie = np.hstack(movie)
        movie2 = np.hstack(movie2)

        contours = measure.find_contours(movie2, 0.5)

        contours = merge_contours(contours, merge_thresh[j])

        cents = [i.mean(axis=0) for i in contours]
        sorted_indices = sort_by_second_element_with_index(cents)
        contours = [contours[x] for x in sorted_indices]

        assert len(contours) == len(classes)

        plt.tight_layout()
        fig.savefig(save_path + f"LDA_{hit}.png", bbox_inches="tight", dpi=300)
        plt.show()

        # Plot movie
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))
        ax.imshow(movie, cmap="gray_r")
        for j, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=colors[classes[j]])
        plt.axis("off")
        fig.savefig(save_path + f"LDA_samples_{hit}.png", bbox_inches="tight", dpi=300)


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
    parser.add_argument("--raw_path", type=str, required=True, help="Path to raw data")
    args = parser.parse_args()

    # Validate that required paths are provided
    if not args.save_path or not args.embeddings_path:
        print("Error: Required arguments are missing.")
        sys.exit(1)

    main(args)

    """
    Example runs for each dataset:

    cellpack dataset
    python src/br/analysis/run_drugdata_analysis.py --save_path "./outputs_npm1_perturb/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/npm1_perturb/" --dataset_name "npm1_perturb" --raw_path "./NPM1_single_cell_drug_perturbations/"
    """
