# Free up cache
import argparse
import gc
import os
import sys

import torch

from br.analysis.analysis_utils import _setup_evaluation_params, config_gpu
from br.models.load_models import get_data_and_models
from br.models.save_embeddings import save_embeddings


def main(args):
    # Free up cache
    gc.collect()
    torch.cuda.empty_cache()

    # Based on the utilization, set the GPU ID
    # Setting a GPU ID is crucial for the script to work well!
    selected_gpu_id_or_uuid = config_gpu()
    selected_gpu_id_or_uuid = "MIG-5c1d3311-7294-5551-9e4f-3535560f5f82"
    import ipdb

    ipdb.set_trace()

    # Set the CUDA_VISIBLE_DEVICES environment variable using the selected ID
    if selected_gpu_id_or_uuid:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu_id_or_uuid
        print(f"CUDA_VISIBLE_DEVICES set to: {selected_gpu_id_or_uuid}")
    else:
        print("No suitable GPU or MIG ID found. Exiting...")

    # Set the device
    device = "cuda:0"

    # # Set working directory and paths
    # os.chdir(args.src_path)

    # Get config path from CYTODL_CONFIG_PATH
    config_path = os.environ.get("CYTODL_CONFIG_PATH")

    # Load data and models
    data_list, all_models, run_names, model_sizes, manifest, _, _ = get_data_and_models(
        args.dataset_name, args.batch_size, config_path + "/results/", args.debug
    )

    # Load evaluation params
    (
        eval_scaled_img,
        eval_scaled_img_params,
        loss_eval_list,
        sample_points_list,
        skew_scale,
    ) = _setup_evaluation_params(manifest, run_names)

    # save embeddings for each model
    save_embeddings(
        args.save_path,
        data_list,
        all_models,
        run_names,
        args.debug,
        ["train", "val", "test"],  # splits to compute embeddings
        device,
        args.meta_key,
        loss_eval_list,
        sample_points_list,
        skew_scale,
        eval_scaled_img,
        eval_scaled_img_params,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for computing embeddings")
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the embeddings."
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
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing.")
    parser.add_argument("--debug", type=bool, default=True, help="Enable debug mode.")

    args = parser.parse_args()

    # Validate that required paths are provided
    if not args.save_path or not args.dataset_name:
        print("Error: Required arguments are missing.")
        sys.exit(1)

    main(args)

    """
    Example run:
    python src/br/analysis/run_embeddings.py --save_path "./outputs/" --sdf False --dataset_name "pcna" --batch_size 5 --debug False
    """