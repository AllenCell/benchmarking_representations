import os
from pathlib import Path
import pandas as pd
import sys
from br.analysis.analysis_utils import (
    _setup_gpu, 
    _latent_walk_save_recons, 
    _dataset_specific_subsetting,
    _archetypes_save_recons, 
    _pseudo_time_analysis
)
from br.models.compute_features import get_embeddings
from br.models.load_models import _load_model_from_path
from br.models.utils import get_all_configs_per_dataset
from br.features.reconstruction import stratified_latent_walk
import argparse
from br.features.archetype import AA_Fast


def main(args):
    _setup_gpu()
    device = "cuda:0"

    config_path = os.environ.get("CYTODL_CONFIG_PATH")
    results_path = config_path + "/results/"

    run_name = "Rotation_invariant_pointcloud_jitter"
    DATASET_INFO = get_all_configs_per_dataset(results_path)
    models = DATASET_INFO[args.dataset_name]
    checkpoints = models["model_checkpoints"]
    checkpoints = [i for i in checkpoints if run_name in i]
    assert len(checkpoints) == 1
    all_ret, df = get_embeddings([run_name], args.dataset_name, DATASET_INFO, args.embeddings_path)
    model, x_label, latent_dim, model_size = _load_model_from_path(checkpoints[0], False, device)

    all_ret, stratify_key, n_archetypes, viz_params = _dataset_specific_subsetting(
        all_ret, args.dataset_name
    )

    # Compute stratified latent walk
    key = "pcloud"
    this_save_path = Path(args.save_path) / Path("latent_walks")
    this_save_path.mkdir(parents=True, exist_ok=True)

    stratified_latent_walk(
        model,
        device,
        all_ret,
        "pcloud",
        256,
        256,
        2,
        this_save_path,
        stratify_key,
        latent_walk_range=[-2, 0, 2],
        z_max=viz_params['z_max'],
        z_ind=viz_params['z_ind'],
    )

    # Save reconstruction plots
    _latent_walk_save_recons(this_save_path, stratify_key, viz_params)

    # Archetype analysis
    matrix = all_ret[[i for i in all_ret.columns if "mu" in i]].values
    aa = AA_Fast(n_archetypes, max_iter=1000, tol=1e-6).fit(matrix)
    archetypes_df = pd.DataFrame(aa.Z, columns=[f"mu_{i}" for i in range(matrix.shape[1])])

    this_save_path = Path(args.save_path) / Path("archetypes")
    this_save_path.mkdir(parents=True, exist_ok=True)

    _archetypes_save_recons(
        model, archetypes_df, device, key, viz_params, this_save_path
    )

    # Pseudotime analysis
    if "volume_of_nucleus_um3" in all_ret.columns:
        _pseudo_time_analysis(model, all_ret, args.save_path, device, key, viz_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for computing embeddings")
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the embeddings."
    )
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
    Example run:
    python src/br/analysis/punctate_analysis.py --save_path "./testing/" --embeddings_path "./morphology_appropriate_representation_learning/model_embeddings/pcna" --dataset_name "pcna"
    """
