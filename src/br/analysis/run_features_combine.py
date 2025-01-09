import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from br.features.plot import collect_outputs, plot
from br.models.utils import get_all_configs_per_dataset


def main(args):

    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # Get config path from CYTODL_CONFIG_PATH
    config_path = os.environ.get("CYTODL_CONFIG_PATH")
    run_names_1 = get_all_configs_per_dataset(config_path + "/results/")[args.dataset_name_1][
        "names"
    ]

    # Polar plot visualization
    # Load saved csvs
    csvs = [i for i in os.listdir(args.feature_path_1) if i.split(".")[-1] == "csv"]
    csvs = [i.split(".")[0] for i in csvs]
    # Remove non metric related csvs
    csvs = [i for i in csvs if i not in run_names_1 and i not in ["image", "pcloud"]]

    for csv in csvs:
        df1 = pd.read_csv(args.feature_path_1 + csv + ".csv")
        df2 = pd.read_csv(args.feature_path_2 + csv + ".csv")
        df2["model"] = df2["model"].apply(lambda x: x + "_2")
        df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
        df.to_csv(args.save_path + csv + ".csv")

    run_names_2 = [i + "_2" for i in run_names_1]
    run_names = run_names_1 + run_names_2
    csvs = [i for i in os.listdir(args.save_path) if i.split(".")[-1] == "csv"]
    csvs = [i.split(".")[0] for i in csvs]
    # Remove non metric related csvs
    csvs = [i for i in csvs if i not in run_names_1 and i not in ["image", "pcloud"]]

    # classification and regression metrics are unique to each dataset
    unique_metrics = [i for i in csvs if "classification" in i or "regression" in i]
    # Collect dataframe and make plots
    df, df_non_agg = collect_outputs(args.save_path, "std", run_names, csvs)
    plot(
        args.save_path,
        df,
        run_names,
        args.dataset_name_1 + "_" + args.dataset_name_2,
        "std",
        unique_metrics,
        df_non_agg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for computing features")
    parser.add_argument(
        "--feature_path_1",
        type=str,
        required=True,
        help="Path to features for dataset 1.",
    )
    parser.add_argument(
        "--feature_path_2",
        type=str,
        required=True,
        help="Path to features for dataset 2.",
    )
    parser.add_argument("--save_path", type=str, required=True, help="Path to save results.")
    parser.add_argument("--dataset_name_1", type=str, required=True, help="Name of the dataset 1.")
    parser.add_argument("--dataset_name_2", type=str, required=True, help="Name of the dataset 2.")
    args = parser.parse_args()
    main(args)

    """
    Example run:
    python src/br/analysis/run_features_combine.py --feature_path_1 './outputs_npm1_remake/' --feature_path_2 './outputs_npm1_64_res_remake/' --save_path "./outputs_npm1_combine/" --dataset_name_1 "npm1" --dataset_name_2 "npm1_64_res"
    """
