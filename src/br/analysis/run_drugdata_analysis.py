import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from br.chandrasekaran_et_al.utils import _plot, perturbation_detection
from br.models.compute_features import get_embeddings
from br.models.utils import get_all_configs_per_dataset


def _get_featurecols(df):
    """returna  list of featuredata columns."""
    return [c for c in df.columns if "mu_" in c]


def _get_featurecols_cellprofiler(df):
    """returna  list of featuredata columns."""
    return [i for i in df.columns if "Mean" in i or "StDev" in i or "Median" in i]


def _get_featuredata(df, get_featurecols_fn):
    """return dataframe of just featuredata columns."""
    return df[get_featurecols_fn(df)]


def main(args):

    config_path = os.environ.get("CYTODL_CONFIG_PATH")
    results_path = config_path + "/results/"

    dataset_name = args.dataset_name
    DATASET_INFO = get_all_configs_per_dataset(results_path)
    dataset = DATASET_INFO[dataset_name]
    run_names = dataset["names"] + ["cellprofiler"]

    all_ret, df = get_embeddings(run_names, args.dataset_name, DATASET_INFO, args.embeddings_path)
    all_ret["well_position"] = "A0"  # dummy
    all_ret["Assay_Plate_Barcode"] = "Plate0"  # dummy

    df_1 = all_ret.loc[~all_ret["model"].isin(["cellprofiler"])].reset_index(drop=True)
    pert1 = perturbation_detection(df_1, _get_featurecols, _get_featuredata)

    df_2 = all_ret.loc[all_ret["model"].isin(["cellprofiler"])].reset_index(drop=True)
    pert2 = perturbation_detection(df_2, _get_featurecols_cellprofiler, _get_featuredata)

    pert = pd.concat([pert1, pert2], axis=0).reset_index(drop=True)

    this_save_path = Path(args.save_path)
    this_save_path.mkdir(parents=True, exist_ok=True)
    _plot(pert, this_save_path, run_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for computing perturbation detection metrics"
    )
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the results.")
    parser.add_argument(
        "--embeddings_path", type=str, required=True, help="Path to the saved embeddings."
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    args = parser.parse_args()

    # Validate that required paths are provided
    if not args.save_path or not args.embeddings_path:
        print("Error: Required arguments are missing.")
        sys.exit(1)

    main(args)

    """
    Example runs for each dataset:

    cellpack dataset
    python src/br/analysis/run_drugdata_analysis.py --save_path "./outputs_npm1_perturb/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/npm1_perturb/" --dataset_name "npm1_perturb"
    """
