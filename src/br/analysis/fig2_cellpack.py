import os
import torch
from Pathlib import Path
import pandas as pd
from br.analysis.analysis_utils import _setup_gpu
from br.models.compute_features import get_embeddings
from br.models.load_models import _load_model_from_path
from br.models.utils import get_all_configs_per_dataset
from br.features.plot import plot_stratified_pc
from br.features.reconstruction import stratified_latent_walk, save_pcloud
import argparse
from br.features.utils import (
    normalize_intensities_and_get_colormap,
    normalize_intensities_and_get_colormap_apply,
)

from br.features.plot import plot_pc_saved
from br.features.archetype import AA_Fast
import numpy as np


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
    all_ret, df = get_embeddings([run_name], args.dataset_name, DATASET_INFO, args.save_path)
    model, x_label, latent_dim, model_size = _load_model_from_path(checkpoints[0], False, device)

    # Compute stratified latent walk
    key = "pcloud"
    stratify_key = "rule"
    z_max = 0.3
    z_ind = 1
    flip = True
    views = ["xy"]
    xlim = [-20, 20]
    ylim = [-20, 20]
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
        z_max=z_max,
        z_ind=z_ind,
    )

    # Save reconstruction plots
    items = os.listdir(this_save_path)
    fnames = [i for i in items if i.split(".")[-1] == "csv"]
    fnames = [i for i in fnames if i.split("_")[1] == "0"]
    names = [i.split(".")[0] for i in fnames]
    cm_name = "inferno"

    all_df = []
    for idx, _ in enumerate(fnames):
        fname = fnames[idx]
        df = pd.read_csv(f"{this_save_path}/{fname}", index_col=0)
        df, cmap, vmin, vmax = normalize_intensities_and_get_colormap(
            df, pcts=[5, 95], cm_name=cm_name
        )
        df[stratify_key] = names[idx]
        all_df.append(df)
    df = pd.concat(all_df, axis=0).reset_index(drop=True)

    plot_stratified_pc(df, xlim, ylim, stratify_key, this_save_path, cmap, flip)

    # Archetype analysis
    # Fit 6 archetypes
    this_ret = all_ret
    matrix = this_ret[[i for i in this_ret.columns if "mu" in i]].values

    n_archetypes = 6
    aa = AA_Fast(n_archetypes, max_iter=1000, tol=1e-6).fit(matrix)
    archetypes_df = pd.DataFrame(aa.Z, columns=[f"mu_{i}" for i in range(matrix.shape[1])])

    this_save_path = Path(args.save_path) / Path("archetypes")
    this_save_path.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    key = "pcloud"
    all_xhat = []
    with torch.no_grad():
        for i in range(n_archetypes):
            z_inf = torch.tensor(archetypes_df.iloc[i].values).unsqueeze(axis=0)
            z_inf = z_inf.to(device)
            z_inf = z_inf.float()
            decoder = model.decoder[key]
            xhat = decoder(z_inf)
            xhat = xhat.detach().cpu().numpy()
            xhat = save_pcloud(xhat[0], this_save_path, i, z_max, z_ind)
            all_xhat.append(xhat)

    names = [str(i) for i in range(n_archetypes)]
    key = "archetype"

    plot_pc_saved(this_save_path, names, key, flip, 0.5, views, xlim, ylim)

    # Save numpy arrays
    key = "archetype"
    items = os.listdir(this_save_path)
    fnames = [i for i in items if i.split(".")[-1] == "csv"]
    names = [i.split(".")[0] for i in fnames]

    df = pd.DataFrame([])
    for idx, _ in enumerate(fnames):
        fname = fnames[idx]
        dft = pd.read_csv(f"{this_save_path}/{fname}", index_col=0)
        dft[key] = names[idx]
        df = pd.concat([df, dft], ignore_index=True)

    archetypes = ["0", "1", "2", "3", "4", "5"]

    for arch in archetypes:
        this_df = df.loc[df["archetype"] == arch].reset_index(drop=True)
        np_arr = this_df[["x", "y", "z"]].values
        np.save(this_save_path / Path(f"{arch}.npy"), np_arr)


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
