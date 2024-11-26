import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from br.analysis.analysis_utils import (
    archetypes_polymorphic,
    archetypes_save_recons,
    dataset_specific_subsetting,
    latent_walk_polymorphic,
    latent_walk_save_recons,
    pseudo_time_analysis,
    setup_gpu,
    str2bool,
)
from br.features.archetype import AA_Fast
from br.features.reconstruction import stratified_latent_walk
from br.models.compute_features import get_embeddings
from br.models.load_models import _load_model_from_path
from br.models.utils import get_all_configs_per_dataset


def main(args):
    setup_gpu()
    device = "cuda:0"

    config_path = os.environ.get("CYTODL_CONFIG_PATH")
    results_path = config_path + "/results/"

    run_name = args.run_name
    DATASET_INFO = get_all_configs_per_dataset(results_path)
    models = DATASET_INFO[args.dataset_name]
    checkpoints = models["model_checkpoints"]
    checkpoints = [i for i in checkpoints if run_name in i]
    assert len(checkpoints) == 1
    all_ret, df = get_embeddings([run_name], args.dataset_name, DATASET_INFO, args.embeddings_path)
    model, x_label, latent_dim, model_size = _load_model_from_path(checkpoints[0], False, device)

    all_ret, stratify_key, n_archetypes, viz_params = dataset_specific_subsetting(
        all_ret, args.dataset_name
    )

    # Compute stratified latent walk
    key = "pcloud"  # all analysis on pointcloud models
    this_save_path = Path(args.save_path) / Path("latent_walks")
    this_save_path.mkdir(parents=True, exist_ok=True)

    if args.sdf:
        latent_walk_polymorphic(stratify_key, all_ret, this_save_path, latent_dim)
    else:
        stratified_latent_walk(
            model,
            device,
            all_ret,
            "pcloud",
            latent_dim,
            latent_dim,
            2,
            this_save_path,
            stratify_key,
            latent_walk_range=[-2, 0, 2],
            z_max=viz_params["z_max"],
            z_ind=viz_params["z_ind"],
        )

        # Save reconstruction plots
        latent_walk_save_recons(this_save_path, stratify_key, viz_params, args.dataset_name)

    # Archetype analysis
    matrix = all_ret[[i for i in all_ret.columns if "mu" in i]].values
    aa = AA_Fast(n_archetypes, max_iter=1000, tol=1e-6).fit(matrix)
    archetypes_df = pd.DataFrame(aa.Z, columns=[f"mu_{i}" for i in range(matrix.shape[1])])

    this_save_path = Path(args.save_path) / Path("archetypes")
    this_save_path.mkdir(parents=True, exist_ok=True)

    if args.sdf:
        archetypes_polymorphic(this_save_path, archetypes_df, all_ret, matrix)
    else:
        archetypes_save_recons(model, archetypes_df, device, key, viz_params, this_save_path)

    # Pseudotime analysis
    if "volume_of_nucleus_um3" in all_ret.columns:
        pseudo_time_analysis(model, all_ret, args.save_path, device, key, viz_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for computing embeddings")
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the embeddings."
    )
    parser.add_argument("--run_name", type=str, required=True, help="Name of model")
    parser.add_argument(
        "--embeddings_path", type=str, required=True, help="Path to the saved embeddings."
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument(
        "--sdf",
        type=str2bool,
        required=True,
        help="boolean indicating whether the model involves SDFs",
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
    python src/br/analysis/run_analysis.py --save_path "./outputs_cellpack/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/cellpack" --dataset_name "cellpack" --run_name "Rotation_invariant_pointcloud_jitter" --sdf False

    pcna dataset
    python src/br/analysis/run_analysis.py --save_path "./outputs_pcna/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/pcna" --dataset_name "pcna" --run_name "Rotation_invariant_pointcloud_jitter" --sdf False

    other punctate structures dataset:
    python src/br/analysis/run_analysis.py --save_path "./outputs_other_punctate/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/other_punctate/" --dataset_name "other_punctate" --run_name "Rotation_invariant_pointcloud_structurenorm" --sdf False

    npm1 dataset:
    python src/br/analysis/run_analysis.py --save_path "./outputs_npm1/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/npm1/" --dataset_name "npm1" --run_name "Rotation_invariant_pointcloud_SDF" --sdf True

    other polymorphic dataset:
    python src/br/analysis/run_analysis.py --save_path "./outputs_other_polymorphic/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/other_polymorphic/" --dataset_name "other_polymorphic" --run_name "Rotation_invariant_pointcloud_SDF" --sdf True
    """
