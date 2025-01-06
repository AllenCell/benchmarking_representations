# Free up cache
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from br.analysis.analysis_utils import (
    get_feature_params,
    setup_evaluation_params,
    setup_gpu,
    str2bool,
)
from br.features.plot import collect_outputs, plot
from br.models.compute_features import compute_features
from br.models.load_models import get_data_and_models
from br.models.save_embeddings import save_emissions


def main(args):
    # Setup GPUs and set the device
    setup_gpu()
    device = "cuda:0"

    # set batch size to 1 for emission stats/features
    batch_size = 1

    # Get config path from CYTODL_CONFIG_PATH
    config_path = os.environ.get("CYTODL_CONFIG_PATH")

    # Load data and models
    (
        data_list,
        all_models,
        run_names,
        model_sizes,
        manifest,
        keys,
        latent_dims,
    ) = get_data_and_models(
        args.dataset_name, batch_size, config_path + "/results/", args.debug
    )
    max_embed_dim = min(latent_dims)

    # make save path directory
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    if not args.skip_features:
        # Save model sizes to CSV
        sizes_ = pd.DataFrame()
        sizes_["model"] = run_names
        sizes_["model_size"] = model_sizes
        sizes_.to_csv(os.path.join(args.save_path, "model_sizes.csv"))

        # Load evaluation params
        (
            eval_scaled_img,
            eval_scaled_img_params,
            loss_eval_list,
            sample_points_list,
            skew_scale,
        ) = setup_evaluation_params(manifest, run_names)

        # Save emission stats for each model
        max_batches = 40
        save_emissions(
            args.save_path,
            data_list,
            all_models,
            run_names,
            max_batches,
            args.debug,
            device,
            loss_eval_list,
            sample_points_list,
            skew_scale,
            eval_scaled_img,
            eval_scaled_img_params,
        )

        # Compute multi-metric benchmarking params
        (
            rot_inv_params,
            compactness_params,
            classification_params,
            evolve_params,
            regression_params,
        ) = get_feature_params(
            config_path + "/results/", args.dataset_name, manifest, keys, run_names
        )

        metric_list = [
            "Rotation Invariance Error",
            "Evolution Energy",
            "Reconstruction",
            "Classification",
            "Compactness",
        ]
        if regression_params["target_cols"]:
            metric_list.append("Regression")

        # Compute multi-metric benchmarking features
        compute_features(
            dataset=args.dataset_name,
            results_path=config_path + "/results/",
            embeddings_path=args.embeddings_path,
            save_folder=args.save_path,
            data_list=data_list,
            all_models=all_models,
            run_names=run_names,
            use_sample_points_list=sample_points_list,
            keys=keys,
            device=device,
            max_embed_dim=max_embed_dim,
            splits_list=["train", "val", "test"],
            compute_embeds=False,
            classification_params=classification_params,
            regression_params=regression_params,
            metric_list=metric_list,
            loss_eval_list=loss_eval_list,
            evolve_params=evolve_params,
            rot_inv_params=rot_inv_params,
            compactness_params=compactness_params,
        )

    # Polar plot visualization
    # Load saved csvs
    csvs = [i for i in os.listdir(args.save_path) if i.split(".")[-1] == "csv"]
    csvs = [i.split(".")[0] for i in csvs]
    # Remove non metric related csvs
    csvs = [i for i in csvs if i not in run_names and i not in keys]
    csvs = [i for i in csvs if i not in ["image", "pcloud"]]
    # classification and regression metrics are unique to each dataset
    unique_metrics = [i for i in csvs if "classification" in i or "regression" in i]
    # Collect dataframe and make plots
    df, df_non_agg = collect_outputs(args.save_path, "std", run_names, csvs)
    plot(
        args.save_path,
        df,
        run_names,
        args.dataset_name,
        "std",
        unique_metrics,
        df_non_agg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for computing features")
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the embeddings."
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to the saved embeddings.",
    )
    parser.add_argument(
        "--meta_key",
        type=str,
        default=None,
        required=False,
        help="Metadata to add to the embeddings aside from CellId",
    )
    parser.add_argument(
        "--sdf",
        type=str2bool,
        required=True,
        help="boolean indicating whether the experiments involve SDFs",
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset."
    )
    parser.add_argument(
        "--debug", type=str2bool, default=False, help="Enable debug mode."
    )
    parser.add_argument(
        "--skip_features",
        type=str2bool,
        default=False,
        help="Boolean indicating whether to skip feature calculation and load pre-computed csvs",
    )

    args = parser.parse_args()

    # Validate that required paths are provided
    if not args.embeddings_path or not args.save_path or not args.dataset_name:
        print("Error: Required arguments are missing.")
        sys.exit(1)

    main(args)

    """
    Example run:
    python src/br/analysis/run_features.py --save_path "./outputs/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/pcna" --sdf False --dataset_name "pcna"

    python src/br/analysis/run_features.py --save_path "/outputs_cellpack/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/cellpack" --sdf False --dataset_name "cellpack" --debug False

    python src/br/analysis/run_features.py --save_path "./outputs_npm1_64_res_remake/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/npm1_64_res" --sdf True --dataset_name "npm1_64_res" --debug False
    """
