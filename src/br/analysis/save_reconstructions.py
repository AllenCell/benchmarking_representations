# Free up cache
import argparse
import os
import sys
from pathlib import Path

from br.analysis.analysis_utils import (
    generate_reconstructions,
    save_supplemental_figure_punctate_reconstructions,
    save_supplemental_figure_sdf_reconstructions,
    setup_gpu,
    str2bool,
)
from br.models.load_models import get_data_and_models

test_ids_per_dataset_ = {
    "cellpack": [
        "9c1ff213-4e9e-4b73-a942-3baf9d37a50f_0",
        "9c1ff213-4e9e-4b73-a942-3baf9d37a50f_1",
        "9c1ff213-4e9e-4b73-a942-3baf9d37a50f_2",
        "9c1ff213-4e9e-4b73-a942-3baf9d37a50f_3",
        "9c1ff213-4e9e-4b73-a942-3baf9d37a50f_4",
        "9c1ff213-4e9e-4b73-a942-3baf9d37a50f_5",
    ],
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
    "other_punctate": ["721646", "873680", "994027", "490385", "451974", "811336", "835431"],
    "npm1": ["964798", "661110", "644401", "967887", "703621"],
    "other_polymorphic": ["691110", "723687", "816468", "800894"],
}


def main(args):
    # Setup GPUs and set the device
    setup_gpu()
    device = "cuda:0"

    # set batch size to 1 for emission stats/features
    batch_size = 1

    # Get config path from CYTODL_CONFIG_PATH
    config_path = os.environ.get("CYTODL_CONFIG_PATH")

    test_ids = args.test_ids
    if not test_ids:
        test_ids = test_ids_per_dataset_[args.dataset_name]

    if args.dataset_name == "cellpack":
        args.debug = True

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

    # make save path directory
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    if args.generate_reconstructions:
        generate_reconstructions(
            all_models, data_list, run_names, keys, test_ids, device, args.save_path
        )

    if args.sdf:
        save_supplemental_figure_sdf_reconstructions(manifest, test_ids, args.save_path)
    else:
        save_supplemental_figure_punctate_reconstructions(
            manifest,
            test_ids,
            run_names,
            args.save_path,
            args.normalize_across_recons,
            args.dataset_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for computing features")
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the embeddings."
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--debug", type=str2bool, default=False, help="Enable debug mode.")
    parser.add_argument(
        "--sdf", type=str2bool, default=True, help="Whether the experiments involve SDFs"
    )
    parser.add_argument(
        "--test_ids", default=False, nargs="+", help="List of test set cellids to reconstruct"
    )
    parser.add_argument(
        "--generate_reconstructions",
        type=str2bool,
        default=False,
        help="Whether to skip generating reconstructions",
    )
    parser.add_argument(
        "--normalize_across_recons",
        type=str2bool,
        default=False,
        help="Whether to normalize across all inputs",
    )

    args = parser.parse_args()

    # Validate that required paths are provided
    if not args.save_path or not args.dataset_name:
        print("Error: Required arguments are missing.")
        sys.exit(1)

    main(args)

    """
    Example run:

    cellPACK dataset
    python src/br/analysis/save_reconstructions.py --save_path "./outputs_cellpack/reconstructions/" --dataset_name "cellpack" --generate_reconstructions True --sdf False

    PCNA dataset
    python src/br/analysis/save_reconstructions.py --save_path "./outputs_pcna/reconstructions/" --dataset_name "pcna" --generate_reconstructions True --sdf False --normalize_across_recons True

    NPM1 dataset
    python src/br/analysis/save_reconstructions.py --save_path "./outputs_npm1/reconstructions/" --dataset_name "npm1" --test_ids 964798 661110 644401 967887 703621 --generate_reconstructions True --sdf True

    Other polymorphic dataset
    python src/br/analysis/save_reconstructions.py --save_path "./outputs_other_polymorphic/reconstructions/" --dataset_name "other_polymorphic" --test_ids 691110 723687 816468 800894 --generate_reconstructions True --sdf True

    Other punctate dataset
    python src/br/analysis/save_reconstructions.py --save_path "./outputs_other_punctate/reconstructions/" --dataset_name "other_punctate" --generate_reconstructions True --normalize_across_recons False --sdf False
    """
