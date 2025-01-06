import argparse
import os
import sys
from pathlib import Path

from br.analysis.analysis_utils import setup_evaluation_params, setup_gpu, str2bool
from br.models.load_models import get_data_and_models
from br.models.save_embeddings import save_embeddings


def main(args):

    # Setup GPUs and set the device
    setup_gpu()
    device = "cuda:0"

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
    ) = setup_evaluation_params(manifest, run_names, args.eval_scaled_img_resolution)

    # make save path directory
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

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
        type=str2bool,
        required=True,
        help="boolean indicating whether the experiments involve SDFs",
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing.")
    parser.add_argument("--debug", type=str2bool, default=True, help="Enable debug mode.")
    parser.add_argument(
        "--eval_scaled_img_resolution",
        type=int,
        default=None,
        required=False,
        help="Resolution for SDF reconstruction",
    )

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
