import gc
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from br.features.plot import plot_pc_saved, plot_stratified_pc
from br.features.reconstruction import save_pcloud
from br.features.utils import (
    normalize_intensities_and_get_colormap,
    normalize_intensities_and_get_colormap_apply,
)
from br.models.utils import get_all_configs_per_dataset


def get_gpu_info():
    # Run nvidia-smi command and get the output
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()


def check_mig():
    # Check if MIG is enabled
    cmd = ["nvidia-smi", "-L"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return "MIG" in result.stdout


def get_mig_ids():
    # Get the MIG UUIDs
    cmd = ["nvidia-smi", "-L"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    mig_ids = []
    for line in result.stdout.splitlines():
        if "MIG" in line:
            mig_id = line.split("(UUID: ")[-1].strip(")")
            mig_ids.append(mig_id)
    return mig_ids


def config_gpu():
    selected_gpu_id_or_uuid = ""
    is_mig = check_mig()

    gpu_info = get_gpu_info()
    lines = gpu_info.splitlines()

    for line in lines:
        index, uuid, name, utilization = map(str.strip, line.split(","))

        # If utilization is [N/A], treat it as less than 10
        if utilization == "[N/A]":
            utilization = -1  # Assign a value less than 10 to simulate "idle"
        else:
            utilization = int(utilization)

        # Check if GPU utilization is under 10% (indicating it's idle)
        if utilization < 10:
            if is_mig:
                mig_ids = get_mig_ids()
                if mig_ids:
                    selected_gpu_id_or_uuid = mig_ids[0]  # Select the first MIG ID
                    break  # Exit the loop after finding the first MIG ID
            else:
                selected_gpu_id_or_uuid = uuid
                print(f"Selected UUID is {selected_gpu_id_or_uuid}")
                break
    return selected_gpu_id_or_uuid


def _setup_gpu():
    # Free up cache
    gc.collect()
    torch.cuda.empty_cache()

    # Based on the utilization, set the GPU ID
    # Setting a GPU ID is crucial for the script to work well!
    selected_gpu_id_or_uuid = config_gpu()
    selected_gpu_id_or_uuid = 'MIG-ffdee303-0dd4-513d-b18c-beba028b49c7'

    # Set the CUDA_VISIBLE_DEVICES environment variable using the selected ID
    if selected_gpu_id_or_uuid:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu_id_or_uuid
        print(f"CUDA_VISIBLE_DEVICES set to: {selected_gpu_id_or_uuid}")
    else:
        print("No suitable GPU or MIG ID found. Exiting...")


def _setup_evaluation_params(manifest, run_names):
    """Return evaluation params related to.

    1. loss_eval_list - which loss to use for each model (Defaults to Chamfer loss)
    2. skew_scale - Hyperparameter associated with sampling of pointclouds from images
    3. sample_points_list - whether to sample pointclouds for each model
    4. eval_scaled_img - whether to scale the images for evaluation (specific to SDF models)
    5. eval_scaled_img_params - parameters like mesh paths, scale factors, pointcloud paths associated
    with evaluating scaled images
    """
    eval_scaled_img = [False] * len(run_names)
    eval_scaled_img_params = [{}] * len(run_names)

    if "SDF" in "\t".join(run_names):
        eval_scaled_img_resolution = 32
        gt_mesh_dir = manifest["mesh_folder"].iloc[0]
        gt_sampled_pts_dir = manifest["pointcloud_folder"].iloc[0]
        gt_scale_factor_dict_path = manifest["scale_factor"].iloc[0]
        eval_scaled_img_params = []
        for name_ in run_names:
            if "seg" in name_:
                model_type = "seg"
            elif "SDF" in name_:
                model_type = "sdf"
            elif "pointcloud" in name_:
                model_type = "iae"
            eval_scaled_img_params.append(
                {
                    "eval_scaled_img_model_type": model_type,
                    "eval_scaled_img_resolution": eval_scaled_img_resolution,
                    "gt_mesh_dir": gt_mesh_dir,
                    "gt_scale_factor_dict_path": gt_scale_factor_dict_path,
                    "gt_sampled_pts_dir": gt_sampled_pts_dir,
                    "mesh_ext": "stl",
                }
            )
        loss_eval_list = [torch.nn.MSELoss(reduction="none")] * len(run_names)
        sample_points_list = [False] * len(run_names)
        skew_scale = None
    else:
        loss_eval_list = None
        skew_scale = 100
        sample_points_list = []
        for name_ in run_names:
            if "image" in name_:
                sample_points_list.append(True)
            else:
                sample_points_list.append(False)
    return eval_scaled_img, eval_scaled_img_params, loss_eval_list, sample_points_list, skew_scale


def _setup_evolve_params(run_names, data_config_list, keys):
    """
    Set up dataloader parameters specific to the evolution energy metric 
    """
    eval_meshed_img = [False] * len(run_names)
    eval_meshed_img_model_type = [None] * len(run_names)
    compute_evolve_dataloaders = True
    if "SDF" in "\t".join(run_names):
        eval_meshed_img = [True] * len(run_names)
        eval_meshed_img_model_type = []
        for name_ in run_names:
            if "seg" in name_:
                model_type = "seg"
            elif "SDF" in name_:
                model_type = "sdf"
            elif "pointcloud" in name_:
                model_type = "iae"
            eval_meshed_img_model_type.append(model_type)

    evolve_params = {
        "modality_list_evolve": keys,
        "config_list_evolve": data_config_list,
        "num_evolve_samples": 40,
        "compute_evolve_dataloaders": compute_evolve_dataloaders,
        "eval_meshed_img": eval_meshed_img,
        "eval_meshed_img_model_type": eval_meshed_img_model_type,
        "skew_scale": 100,
        "only_embedding": False,
        "fit_pca": False,
        "pc_is_iae": False,
    }
    return evolve_params


def _get_feature_params(results_path, dataset_name, manifest, keys, run_names):
    """
    Get parameters associated with calculation of
    1. Rot invariance 
    2. Compactness
    3. Classification
    4. Evolution/Interpolation distance
    5. Regression
    """
    DATA_LIST = get_all_configs_per_dataset(results_path)
    data_config_list = DATA_LIST[dataset_name]["data_paths"]
    cytodl_config_path = os.environ.get("CYTODL_CONFIG_PATH")
    data_config_list = [cytodl_config_path + i for i in data_config_list]
    class_label = DATA_LIST[dataset_name]["classification_label"]
    regression_label = DATA_LIST[dataset_name]["regression_label"]
    evolve_params = _setup_evolve_params(run_names, data_config_list, keys)
    classification_params = {"class_labels": class_label, "df_feat": manifest}
    rot_inv_params = {"squeeze_2d": False, "id": "cell_id", "max_batches": 4000}
    regression_params = {
        "df_feat": manifest,
        "target_cols": regression_label,
        "feature_df_path": None,
    }
    compactness_params = {
        "method": "mle",
        "num_PCs": None,
        "blobby_outlier_max_cc": None,
        "check_duplicates": True,
    }
    return (
        rot_inv_params,
        compactness_params,
        classification_params,
        evolve_params,
        regression_params,
    )


def _dataset_specific_subsetting(all_ret, dataset_name):
    """
    Subset each dataset for analysis. 
    E.g. For PCNA dataset, only look at interphase.
    Also specify dataset specific visualization params
    - z_max (Max value of z at which to slice data)
    - z_ind (Which index is Z - 1, 2, 3)
    - views = ['xy'] (show xy projection)
    - xlim, ylim = [-20, 20] (max scaling for visualization max projection)
    - flip = True when Z and Y are swapped
    """
    if dataset_name == "pcna":
        interphase_stages = [
            "G1",
            "earlyS",
            "earlyS-midS",
            "midS",
            "midS-lateS",
            "lateS",
            "lateS-G2",
            "G2",
        ]
        all_ret = all_ret.loc[all_ret["cell_stage_fine"].isin(interphase_stages)].reset_index(
            drop=True
        )
        stratify_key = "cell_stage_fine"
        viz_params = {"z_max": 0.3, "z_ind": 2, "flip": False}
        n_archetypes = 8
    elif dataset_name == "cellpack":
        stratify_key = "rule"
        viz_params = {"z_max": 0.3, "z_ind": 1, "flip": True}
        n_archetypes = 6
    elif dataset_name == "other_punctate":
        structs = ["NUP153", "SON", "HIST1H2BJ", "SMC1A", "CETN2", "SLC25A17", "RAB5A"]
        all_ret = all_ret.loc[all_ret["structure_name"].isin(structs)].reset_index(drop=True)
        stratify_key = "structure_name"
        viz_params = {"z_max": None, "z_ind": 2, "flip": False}
        n_archetypes = 7
    else:
        raise ValueError("Dataset not in pre-configured list")
    viz_params["views"] = ["xy"]
    viz_params["xlim"] = [-20, 20]
    viz_params["ylim"] = [-20, 20]
    return all_ret, stratify_key, n_archetypes, viz_params


def _latent_walk_save_recons(this_save_path, stratify_key, viz_params):
    """
    Visualize saved latent walks from csvs
    this_save_path - folder where csvs are saved
    stratify_key - metadata by which PCs are stratified (e.g. "rule")
    viz_params - parameters associated with visualization (e.g. xlims, ylims)
    """
    items = os.listdir(this_save_path)
    fnames = [i for i in items if i.split(".")[-1] == "csv"] # get csvs
    fnames = [i for i in fnames if i.split("_")[1] == "0"] # get 1st PC
    names = [i.split(".")[0] for i in fnames]
    cm_name = "YlGnBu"
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

    plot_stratified_pc(
        df,
        viz_params["xlim"],
        viz_params["ylim"],
        stratify_key,
        this_save_path,
        cmap,
        viz_params["flip"],
    )


def _archetypes_save_recons(model, archetypes_df, device, key, viz_params, this_save_path):
    """
    Visualize saved archetypes from archetype matrix dataframe
    """
    all_xhat = []
    with torch.no_grad():
        for i in range(len(archetypes_df)):
            z_inf = torch.tensor(archetypes_df.iloc[i].values).unsqueeze(axis=0)
            z_inf = z_inf.to(device)
            z_inf = z_inf.float()
            decoder = model.decoder[key]
            xhat = decoder(z_inf)
            xhat = xhat.detach().cpu().numpy()
            xhat = save_pcloud(
                xhat[0], this_save_path, i, viz_params["z_max"], viz_params["z_ind"]
            )
            all_xhat.append(xhat)

    names = [str(i) for i in range(len(archetypes_df))]
    key = "archetype"
    plot_pc_saved(
        this_save_path,
        names,
        key,
        viz_params["flip"],
        0.5,
        viz_params["views"],
        viz_params["xlim"],
        viz_params["ylim"],
    )

    # Save numpy arrays for mitsuba visualization
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

    for arch in names:
        this_df = df.loc[df["archetype"] == arch].reset_index(drop=True)
        np_arr = this_df[["x", "y", "z"]].values
        np.save(this_save_path / Path(f"{arch}.npy"), np_arr)


def _pseudo_time_analysis(model, all_ret, save_path, device, key, viz_params, bins=None):
    """
    Psuedotime analysis for PCNA and NPM1 dataset
    """
    if not bins:
        # Pseudotime bins based on npm1 dataset from WTC-11 hIPS single cell image dataset
        bins = [
            (247.407, 390.752),
            (390.752, 533.383),
            (533.383, 676.015),
            (676.015, 818.646),
            (818.646, 961.277),
        ]
    correct_bins = []
    for ind, row in all_ret.iterrows():
        this_bin = []
        for bin_ in bins:
            if (row["volume_of_nucleus_um3"] > bin_[0]) and (
                row["volume_of_nucleus_um3"] <= bin_[1]
            ):
                this_bin.append(bin_)
        if row["volume_of_nucleus_um3"] < bins[0][0]:
            this_bin.append(bin_)
        if row["volume_of_nucleus_um3"] > bins[4][1]:
            this_bin.append(bin_)
        assert len(this_bin) == 1
        correct_bins.append(this_bin[0])
    all_ret["vol_bins"] = correct_bins
    all_ret["vol_bins_inds"] = pd.factorize(all_ret["vol_bins"])[0]

    # Save reconstructions per bin
    this_save_path = Path(save_path) / Path("pseudo_time")
    this_save_path.mkdir(parents=True, exist_ok=True)

    cols = [i for i in all_ret.columns if "mu" in i]
    for ind, gr in all_ret.groupby(["vol_bins"]):
        this_stage_df = gr.reset_index(drop=True)
        this_stage_mu = this_stage_df[cols].values
        mean_mu = this_stage_mu.mean(axis=0)
        dist = (this_stage_mu - mean_mu) ** 2
        dist = np.sum(dist, axis=1)
        z_inf = torch.tensor(mean_mu).unsqueeze(axis=0)
        z_inf = z_inf.to(device)
        z_inf = z_inf.float()

        decoder = model.decoder["pcloud"]
        xhat = decoder(z_inf)
        xhat = save_pcloud(
            xhat[0], this_save_path, str(ind), viz_params["z_max"], viz_params["z_ind"]
        )

    names = os.listdir(this_save_path)
    names = [i for i in names if i.split(".")[-1] == "csv"]
    names = [i.split(".csv")[0] for i in names]
    plot_pc_saved(
        this_save_path,
        names,
        key,
        viz_params["flip"],
        0.5,
        viz_params["views"],
        viz_params["xlim"],
        viz_params["ylim"],
    )
