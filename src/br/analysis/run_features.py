# Free up cache
import argparse
import gc
import os
import sys

import pandas as pd
import torch

from br.analysis.analysis_utils import (
    _get_feature_params,
    _setup_evaluation_params,
    config_gpu,
)
from br.models.compute_features import compute_features
from br.models.load_models import get_data_and_models
from br.models.save_embeddings import save_emissions


def main(args):
    # Free up cache
    gc.collect()
    torch.cuda.empty_cache()

    # Based on the utilization, set the GPU ID
    # Setting a GPU ID is crucial for the script to work well!
    selected_gpu_id_or_uuid = config_gpu()
    selected_gpu_id_or_uuid = "MIG-5c1d3311-7294-5551-9e4f-3535560f5f82"

    # Set the CUDA_VISIBLE_DEVICES environment variable using the selected ID
    if selected_gpu_id_or_uuid:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu_id_or_uuid
        print(f"CUDA_VISIBLE_DEVICES set to: {selected_gpu_id_or_uuid}")
    else:
        print("No suitable GPU or MIG ID found. Exiting...")

    # Set the device
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
    ) = get_data_and_models(args.dataset_name, batch_size, config_path + "/results/", args.debug)
    max_embed_dim = min(latent_dims)

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
    ) = _setup_evaluation_params(manifest, run_names)

    # Save emission stats for each model
    args.debug = True
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
    ) = _get_feature_params(
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for computing features")
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the embeddings."
    )
    parser.add_argument(
        "--embeddings_path", type=str, required=True, help="Path to the saved embeddings."
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
        type=bool,
        required=True,
        help="boolean indicating whether the experiments involve SDFs",
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--debug", type=bool, default=False, help="Enable debug mode.")

    args = parser.parse_args()

    # Validate that required paths are provided
    if not args.embeddings_path or not args.save_path or not args.dataset_name:
        print("Error: Required arguments are missing.")
        sys.exit(1)

    main(args)

    """
    Example run:
    python src/br/analysis/run_features.py --save_path "./testing/" --embeddings_path "/allen/aics/modeling/ritvik/projects/second_clones/benchmarking_representations/test_pcna_save_embeddings_revisit/" --sdf False --dataset_name "pcna"
    """
