import argparse
import gc
import os
import subprocess
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mesh_to_sdf
import numpy as np
import pandas as pd
import pyvista as pv
import torch
import trimesh
import yaml
from aicsimageio import AICSImage
from sklearn.decomposition import PCA
from tqdm import tqdm
import random

from br.data.utils import get_iae_reconstruction_3d_grid
from br.features.plot import plot_pc_saved, plot_stratified_pc
from br.features.reconstruction import save_pcloud
from br.features.utils import (
    normalize_intensities_and_get_colormap,
    normalize_intensities_and_get_colormap_apply,
)
from br.models.utils import get_all_configs_per_dataset, move


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_gpu_info():
    # Run nvidia-smi command and get the output
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # print(result)
    return result.stdout.strip()


def check_mig():
    # Check if MIG is enabled
    cmd = ["nvidia-smi", "-L"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return "MIG" in result.stdout


def get_mig_ids(gpu_uuid):
    try:
        # Get the list of GPUs
        output = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=,index,uuid", "--format=csv,noheader"]
            )
            .decode("utf-8")
            .strip()
            .split("\n")
        )

        # Find the index of the specified GPU UUID
        gpu_index = -1
        for i, line in enumerate(output):
            if gpu_uuid in line:
                gpu_index = i
                break

        if gpu_index == -1:
            print(f"GPU UUID {gpu_uuid} not found.")
            return []

        # Now we need to get the MIG IDs for this GPU
        mig_ids = []
        # Run nvidia-smi command to get detailed information including MIG IDs
        detailed_output = (
            subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8").strip().split("\n")
        )

        # Flag to determine if we are in the right GPU section
        in_gpu_section = False
        for line in detailed_output:
            if f"GPU {gpu_index}:" in line:  # Adjusted to check for the specific GPU section
                in_gpu_section = True
            elif "GPU" in line and in_gpu_section:  # Encounter another GPU section
                break

            if in_gpu_section:
                # Check for MIG devices
                if "MIG" in line:
                    mig_id = (
                        line.split("(")[1].split(")")[0].split(" ")[-1]
                    )  # Assuming format is '.... MIG (UUID) ...'
                    mig_ids.append(mig_id.strip())

        return mig_ids

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return []


def config_gpu():
    selected_gpu_id_or_uuid = ""
    is_mig = check_mig()

    gpu_info = get_gpu_info()
    lines = gpu_info.splitlines()

    for line in lines:
        index, uuid, name, mem_used, mem_total = map(str.strip, line.split(","))
        utilization = float(mem_used) * 100 / float(mem_total)

        # Check if GPU utilization is under 20% (indicating it's idle)
        if utilization < 20:
            if is_mig:
                mig_ids = get_mig_ids(uuid)

                if mig_ids:
                    mid_id_rand = random.randint(0, len(mig_ids) - 1)
                    selected_gpu_id_or_uuid = mig_ids[mid_id_rand]  # Select the MIG ID
                    break  # Exit the loop after finding the MIG ID
            else:
                selected_gpu_id_or_uuid = uuid
                print(f"Selected UUID is {selected_gpu_id_or_uuid}")
                break
    return selected_gpu_id_or_uuid


def setup_gpu():
    # Free up cache
    gc.collect()
    torch.cuda.empty_cache()

    # Based on the utilization, set the GPU ID
    # Setting a GPU ID is crucial for the script to work!
    selected_gpu_id_or_uuid = config_gpu()

    # Set the CUDA_VISIBLE_DEVICES environment variable using the selected ID
    if selected_gpu_id_or_uuid:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu_id_or_uuid
        print(f"CUDA_VISIBLE_DEVICES set to: {selected_gpu_id_or_uuid}")
    else:
        print("No suitable GPU or MIG ID found. Exiting...")


def setup_evaluation_params(manifest, run_names):
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


def setup_evolve_params(run_names, data_config_list, keys):
    """Set up dataloader parameters specific to the evolution energy metric."""
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


def get_feature_params(results_path, dataset_name, manifest, keys, run_names):
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
    evolve_params = setup_evolve_params(run_names, data_config_list, keys)
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


def dataset_specific_subsetting(all_ret, dataset_name):
    """Subset each dataset for analysis. E.g. For PCNA dataset, only look at interphase. Also
    specify dataset specific visualization params.

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
        all_ret = all_ret.loc[all_ret["structure_name"].isin(structs)]
        all_ret = all_ret.loc[all_ret["cell_stage"].isin(["M0"])].reset_index(drop=True)
        stratify_key = "structure_name"
        viz_params = {"z_max": None, "z_ind": 2, "flip": False, "structs": structs}
        n_archetypes = 7
    elif dataset_name == "npm1":
        stratify_key = "STR_connectivity_cc_thresh"
        n_archetypes = 5
        viz_params = {}
    elif dataset_name == "other_polymorphic":
        stratify_key = "structure_name"
        structs = ["NPM1", "FBL", "LAMP1", "ST6GAL1"]
        all_ret = all_ret.loc[all_ret["structure_name"].isin(structs)]
        n_archetypes = 4
        viz_params = {}
    else:
        raise ValueError("Dataset not in pre-configured list")
    viz_params["views"] = ["xy"]
    viz_params["xlim"] = [-20, 20]
    viz_params["ylim"] = [-20, 20]
    return all_ret, stratify_key, n_archetypes, viz_params


def viz_other_punctate(this_save_path, viz_params, stratify_key):
    # Norms based on Viana 2023
    # norms used for model training
    model_norms = "./src/br/data/preprocessing/pc_preprocessing/model_structnorms.yaml"
    with open(model_norms) as stream:
        model_norms = yaml.safe_load(stream)

    # norms used for viz
    viz_norms = "./src/br/data/preprocessing/pc_preprocessing/viz_structnorms.yaml"
    with open(viz_norms) as stream:
        viz_norms = yaml.safe_load(stream)

    items = os.listdir(this_save_path)
    for struct in viz_params["structs"]:
        fnames = [i for i in items if i.split(".")[-1] == "csv"]
        fnames = [i for i in fnames if i.split("_")[1] == "0"]
        fnames = [i for i in fnames if i.split("_")[0] in [struct]]
        names = [i.split(".")[0] for i in fnames]

        renorm = model_norms[struct]
        this_viz_norm = viz_norms[struct]
        use_vmin = this_viz_norm[0]
        use_vmax = this_viz_norm[1]

        all_df = []
        for idx, _ in enumerate(fnames):
            fname = fnames[idx]
            df = pd.read_csv(f"{this_save_path}/{fname}", index_col=0)
            df["s"] = df["s"] / 10  # scalar values were scaled by 10 during training
            df["s"] = df["s"] * (renorm[1] - renorm[0]) + renorm[0]  # use model norms
            df[stratify_key] = names[idx]
            all_df.append(df)
        df = pd.concat(all_df, axis=0).reset_index(drop=True)
        if struct in ["NUP153", "SON", "HIST1H2BJ", "SMC1A"]:
            df = df.loc[df["z"] < 0.2].reset_index(drop=True)
        df = normalize_intensities_and_get_colormap_apply(df, use_vmin, use_vmax)
        cmap = plt.get_cmap("YlGnBu")
        plot_stratified_pc(
            df,
            viz_params["xlim"],
            viz_params["ylim"],
            stratify_key,
            this_save_path,
            cmap,
            viz_params["flip"],
        )

        for pc_bin in df[stratify_key].unique():
            this_df = df.loc[df[stratify_key] == pc_bin].reset_index(drop=True)
            np_arr = this_df[["x", "y", "z"]].values
            colors = cmap(this_df["inorm"].values)[:, :3]
            np_arr2 = colors
            np_arr = np.concatenate([np_arr, np_arr2], axis=1)
            np.save(this_save_path / Path(f"{stratify_key}_{pc_bin}.npy"), np_arr)
            cmap = plt.get_cmap("YlGnBu")


def latent_walk_save_recons(this_save_path, stratify_key, viz_params, dataset_name):
    """Visualize saved latent walks from csvs.

    this_save_path - folder where csvs are saved
    stratify_key - metadata by which PCs are stratified (e.g. "rule")
    viz_params - parameters associated with visualization (e.g. xlims, ylims)
    """
    if dataset_name == "other_punctate":
        return viz_other_punctate(this_save_path, viz_params, stratify_key)

    items = os.listdir(this_save_path)
    fnames = [i for i in items if i.split(".")[-1] == "csv"]  # get csvs
    fnames = [i for i in fnames if i.split("_")[1] == "0"]  # get 1st PC
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

    df, cmap, vmin, vmax = normalize_intensities_and_get_colormap(
        df, pcts=[5, 95], cm_name="YlGnBu"
    )

    for idx, _ in enumerate(fnames):
        fname = fnames[idx]
        df = pd.read_csv(f"{this_save_path}/{fname}", index_col=0)
        this_name = names[idx]
        if "s" in df.columns:
            df = normalize_intensities_and_get_colormap_apply(df, vmin, vmax)
            np_arr = df[["x", "y", "z"]].values
            colors = cmap(df["inorm"].values)[:, :3]
            np_arr2 = colors
            np_arr = np.concatenate([np_arr, np_arr2], axis=1)
        else:
            np_arr = df[["x", "y", "z"]].values
        np.save(this_save_path / Path(f"{this_name}.npy"), np_arr)


def archetypes_save_recons(model, archetypes_df, device, key, viz_params, this_save_path):
    """Visualize saved archetypes from archetype matrix dataframe."""
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


def pseudo_time_analysis(model, all_ret, save_path, device, key, viz_params, bins=None):
    """Psuedotime analysis for PCNA and NPM1 dataset."""
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


def latent_walk_polymorphic(stratify_key, all_ret, this_save_path, latent_dim):
    lw_dict = {stratify_key: [], "PC": [], "bin": [], "CellId": []}
    mesh_folder = all_ret["mesh_folder"].iloc[0]  # mesh folder
    for strat in all_ret[stratify_key].unique():
        this_sub_m = all_ret.loc[all_ret[stratify_key] == strat].reset_index(drop=True)
        all_features = this_sub_m[[i for i in this_sub_m.columns if "mu" in i]].values
        dim_size = latent_dim
        pca = PCA(n_components=dim_size)
        pca_features = pca.fit_transform(all_features)
        pca_std_list = pca_features.std(axis=0)
        for rank in [0, 1]:
            latent_walk_range = [-2, 0, 2]
            for value_index, value in enumerate(
                tqdm(latent_walk_range, total=len(latent_walk_range))
            ):
                z_inf = torch.zeros(1, dim_size)
                z_inf[:, rank] += value * pca_std_list[rank]
                z_inf = pca.inverse_transform(z_inf)

                dist = (all_features - z_inf) ** 2
                dist = np.sum(dist, axis=1)
                closest_idx = np.argmin(dist)
                closest_real_id = this_sub_m.iloc[closest_idx]["CellId"]
                mesh = pv.read(mesh_folder + str(closest_real_id) + ".stl")
                mesh.save(this_save_path / Path(f"{strat}_{rank}_{value_index}.ply"))

                lw_dict[stratify_key].append(strat)
                lw_dict["PC"].append(rank)
                lw_dict["bin"].append(value_index)
                lw_dict["CellId"].append(closest_real_id)
    lw_dict = pd.DataFrame(lw_dict)
    lw_dict.to_csv(this_save_path / "latent_walk.csv")


def archetypes_polymorphic(this_save_path, archetypes_df, all_ret, all_features):
    arch_dict = {"CellId": [], "archetype": []}
    mesh_folder = all_ret["mesh_folder"].iloc[0]  # mesh folder
    for i in range(len(archetypes_df)):
        this_mu = archetypes_df.iloc[i].values
        dist = (all_features - this_mu) ** 2
        dist = np.sum(dist, axis=1)
        closest_idx = np.argmin(dist)
        closest_real_id = all_ret.iloc[closest_idx]["CellId"]
        mesh = pv.read(mesh_folder + str(closest_real_id) + ".stl")
        mesh.save(this_save_path / Path(f"{i}.ply"))
        arch_dict["archetype"].append(i)
        arch_dict["CellId"].append(closest_real_id)
    arch_dict = pd.DataFrame(arch_dict)
    arch_dict.to_csv(this_save_path / "archetypes.csv")


def generate_reconstructions(all_models, data_list, run_names, keys, test_ids, device, save_path):
    with torch.no_grad():
        for j, model in enumerate(all_models):
            this_data = data_list[j]
            this_run_name = run_names[j]
            this_key = keys[j]
            for batch in this_data.test_dataloader():
                canonical = None
                if not isinstance(batch["cell_id"], list):
                    if isinstance(batch["cell_id"], torch.Tensor):
                        cell_id = str(batch["cell_id"].item())
                    else:
                        cell_id = str(batch["cell_id"])
                else:
                    cell_id = str(batch["cell_id"][0])
                if cell_id in test_ids:
                    input = batch[this_key].detach().cpu().numpy().squeeze()
                    if "pointcloud_SDF" in this_run_name:
                        eval_scaled_img_resolution = 32
                        uni_sample_points = get_iae_reconstruction_3d_grid(
                            bb_min=-0.5,
                            bb_max=0.5,
                            resolution=eval_scaled_img_resolution,
                            padding=0.1,
                        )
                        uni_sample_points = uni_sample_points.unsqueeze(0).repeat(
                            batch[this_key].shape[0], 1, 1
                        )
                        batch["points"] = uni_sample_points
                        xhat, z, z_params = model(
                            move(batch, device), decode=True, inference=True, return_params=True
                        )
                        recon = xhat[this_key].detach().cpu().numpy().squeeze()
                        recon = recon.reshape(
                            eval_scaled_img_resolution,
                            eval_scaled_img_resolution,
                            eval_scaled_img_resolution,
                        )
                    elif ("pointcloud" in this_run_name) and ("SDF" not in this_run_name):
                        batch = move(batch, device)
                        z, z_params = model.get_embeddings(batch, inference=True)
                        xhat = model.decode_embeddings(
                            z_params, batch, decode=True, return_canonical=True
                        )
                        recon = xhat[this_key].detach().cpu().numpy().squeeze()
                        canonical = recon
                        if "canonical" in xhat.keys():
                            canonical = xhat["canonical"].detach().cpu().numpy().squeeze()
                    else:
                        z = model.encode(move(batch, device))
                        xhat = model.decode(z, return_canonical=True)
                        recon = xhat[this_key].detach().cpu().numpy().squeeze()
                        canonical = xhat["canonical"].detach().cpu().numpy().squeeze()

                    this_save_path = Path(save_path) / Path(this_run_name)
                    this_save_path.mkdir(parents=True, exist_ok=True)
                    np.save(this_save_path / Path(f"{cell_id}.npy"), recon)

                    this_save_path_input = Path(save_path) / Path(this_run_name) / Path("input")
                    this_save_path_input.mkdir(parents=True, exist_ok=True)
                    np.save(this_save_path_input / Path(f"{cell_id}.npy"), input)
                    
                    if canonical is not None:
                        this_save_path_canon = (
                            Path(save_path) / Path(this_run_name) / Path("canonical")
                        )
                        this_save_path_canon.mkdir(parents=True, exist_ok=True)
                        np.save(this_save_path_canon / Path(f"{cell_id}.npy"), canonical)


def save_supplemental_figure_punctate_reconstructions(
    df, test_ids, run_names, reconstructions_path, normalize_across_recons, dataset_name
):
    def slice_(img, slices=None, z_ind=0):
        if not slices:
            return img.max(z_ind)
        mid_z = int(img.shape[0] / 2)
        if z_ind == 0:
            img = img[mid_z - slices : mid_z + slices].max(0)
        if z_ind == 2:
            img = img[:, :, mid_z - slices : mid_z + slices].max(2)
        return img

    def slice_points_(points, z_max, z_loc=0):
        inds = np.where(points[:, z_loc] < z_max)[0]
        points = points[inds, :]
        inds = np.where(points[:, z_loc] > -z_max)[0]
        points = points[inds, :]
        return points

    def _plot_image(input, recon, recon_canonical, dataset_name):
        num_slice = 8

        if dataset_name != "cellpack":
            z_ind = 0
        else:
            z_ind = 2

        input = slice_(input, num_slice, z_ind)
        recon = slice_(recon, num_slice, z_ind)
        recon_canonical = slice_(recon_canonical, num_slice, 2)

        if dataset_name == "cellpack":
            recon = recon.T
            recon_canonical = recon_canonical.T

        i = 2
        fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 4))
        ax.imshow(input, cmap="gray_r")
        ax1.imshow(recon, cmap="gray_r")
        ax2.imshow(recon_canonical, cmap="gray_r")

        ax.set_xticks([])
        ax.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        # max_size = 192
        max_size = recon_canonical.shape[1]
        ax.set_aspect("equal", adjustable="box")
        ax1.set_aspect("equal", adjustable="box")
        ax2.set_aspect("equal", adjustable="box")

        # max_size = 6
        ax.set_ylim([0, max_size])
        ax1.set_ylim([0, max_size])
        ax2.set_ylim([0, max_size])
        ax.set_xlim([0, max_size])
        ax1.set_xlim([0, max_size])
        ax2.set_xlim([0, max_size])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["left"].set_visible(False)

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.spines["left"].set_visible(False)

        ax.set_title("Input")
        ax1.set_title("Reconstruction")
        ax2.set_title("Canonical Reconstruction")
        fig.subplots_adjust(wspace=0, hspace=0)
        return fig

    def _plot_pc(input, recon, recon_canonical, struct, cmap, vmin, vmax, dataset_name):
        z_max = 0.3
        max_size = 15

        if dataset_name != "cellpack":
            z_ind = 2
            canon_z_ind = 2
            if (struct == "pcna") and ((recon == recon_canonical).all() == False):
                canon_z_ind = 1
        else:
            z_ind = 0
            canon_z_ind = 1
        xy_inds = [i for i in [0, 1, 2] if i != z_ind]
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        for index_, this_p in enumerate([input, recon, recon_canonical]):
            if struct in ["NUP153", "HIST1H2BJ", "SMC1A", "SON", "pcna"]:
                if index_ == 2:
                    this_p = slice_points_(this_p, z_max, canon_z_ind)
                else:
                    this_p = slice_points_(this_p, z_max, z_ind)
            if this_p.shape[-1] == 3:
                if (index_ == 2) and (canon_z_ind != z_ind):
                    xy_inds = [i for i in [0, 1, 2] if i != canon_z_ind]
                axes[index_].scatter(
                    this_p[:, xy_inds[0]], this_p[:, xy_inds[1]], c="black", s=2, alpha=0.5
                )
            else:
                if not cmap:
                    this_df = pd.DataFrame(input, columns=["x", "y", "z", "s"])
                    all_df_input, cmap, vmin, vmax = normalize_intensities_and_get_colormap(
                        df=this_df, pcts=[5, 95]
                    )
                this_p = pd.DataFrame(this_p, columns=["x", "y", "z", "s"])
                this_p = normalize_intensities_and_get_colormap_apply(this_p, vmin, vmax)

                if (index_ == 2) and (canon_z_ind != z_ind):
                    xy_inds = [i for i in [0, 1, 2] if i != canon_z_ind]

                x_vals = this_p.iloc[:, xy_inds[0]].values
                y_vals = this_p.iloc[:, xy_inds[1]].values

                axes[index_].scatter(
                    x_vals,
                    y_vals,
                    c=cmap(this_p["inorm"].values),
                    s=2,
                    alpha=0.5,
                )
            axes[index_].spines["top"].set_visible(False)
            axes[index_].spines["right"].set_visible(False)
            axes[index_].spines["bottom"].set_visible(False)
            axes[index_].spines["left"].set_visible(False)
            axes[index_].set_aspect("equal", adjustable="box")
            axes[index_].set_ylim([-max_size, max_size])
            axes[index_].set_xlim([-max_size, max_size])
            axes[index_].set_yticks([])
            axes[index_].set_xticks([])

        axes[0].set_title("Input")
        axes[1].set_title("Reconstruction")
        axes[2].set_title("Canonical Reconstruction")
        return fig

    for m in run_names:
        for i, this_id in enumerate(test_ids):
            struct = "pcna"
            if "structure_name" in df.columns:
                df["CellId"] = df["CellId"].astype(str)
                struct = df.loc[df["CellId"] == this_id]["structure_name"].iloc[0]
            row_index = i

            input_path = reconstructions_path + f"{m}/input/{this_id}.npy"
            input = np.load(input_path).squeeze()

            recon_path = reconstructions_path + f"{m}/{this_id}.npy"
            recon = np.load(recon_path).squeeze()

            recon_path = reconstructions_path + f"{m}/canonical/{this_id}.npy"
            recon_canonical = np.load(recon_path).squeeze()

            if "image" in m:
                fig = _plot_image(input, recon, recon_canonical, dataset_name)
            else:
                cmap = None
                vmin = None
                vmax = None
                if normalize_across_recons:
                    all_df_input = []
                    for c in test_ids:
                        input_path_ = reconstructions_path + f"{m}/input/{this_id}.npy"
                        input_tmp = np.load(input_path_).squeeze()
                        if input.shape[-1] == 4:
                            this_df = pd.DataFrame(input_tmp, columns=["x", "y", "z", "s"])
                            all_df_input.append(this_df)
                    if len(all_df_input) > 0:
                        all_df_input = pd.concat(all_df_input, axis=0).reset_index(drop=True)
                        _, cmap, vmin, vmax = normalize_intensities_and_get_colormap(
                            df=all_df_input, pcts=[5, 95]
                        )
                fig = _plot_pc(
                    input, recon, recon_canonical, struct, cmap, vmin, vmax, dataset_name
                )

            this_save_path_ = Path(reconstructions_path) / Path(m)
            print(this_save_path_, this_id)
            fig.savefig(
                this_save_path_ / Path(f"sample_recons_{this_id}.pdf"),
                bbox_inches="tight",
                dpi=300,
            )
            fig.savefig(
                this_save_path_ / Path(f"sample_recons_{this_id}.png"),
                bbox_inches="tight",
                dpi=300,
            )


def save_supplemental_figure_sdf_reconstructions(df, test_ids, reconstructions_path):
    import pyvista as pv

    pv.start_xvfb()
    gt_test_sdfs = []
    gt_test_segs = []
    for tid in test_ids:
        path = df[df["CellId"] == int(tid)]["sdf_path"].values[0]
        sdf = np.load(path)
        path = df[df["CellId"] == int(tid)]["seg_path"].values[0]
        seg = np.load(path)
        gt_test_sdfs.append(sdf)
        gt_test_segs.append(seg)

    gt_orig_struc = []
    gt_orig_cell = []
    gt_orig_nuc = []
    for tid in test_ids:
        path = df[df["CellId"] == int(tid)]["crop_seg_masked"].values[0]
        seg = AICSImage(path).data.squeeze()
        path = df[df["CellId"] == int(tid)]["crop_seg"].values[0]
        img = AICSImage(path).data.squeeze()
        gt_orig_struc.append(seg)
        gt_orig_nuc.append(img[0])
        gt_orig_cell.append(img[1])

    eval_scaled_img_resolution = 32
    mid_slice_ = int(eval_scaled_img_resolution / 2)
    uni_sample_points_grid = get_iae_reconstruction_3d_grid(
        bb_min=-0.5, bb_max=0.5, resolution=eval_scaled_img_resolution, padding=0.1
    )
    gt_test_sdfs_iae = []
    mesh_folder = df["mesh_folder"].iloc[0]
    for tid in test_ids:
        path = mesh_folder + str(tid) + ".stl"
        mesh = trimesh.load(path)
        bbox = mesh.bounding_box.bounds
        loc = (bbox[0] + bbox[1]) / 2
        scale_factor = (bbox[1] - bbox[0]).max()
        mesh = mesh.apply_translation(-loc)
        mesh = mesh.apply_scale(1 / scale_factor)
        sdf_vals = mesh_to_sdf.mesh_to_sdf(mesh, query_points=uni_sample_points_grid.numpy())
        gt_test_sdfs_iae.append(sdf_vals)

    cmap_inverted = mcolors.ListedColormap(np.flipud(plt.cm.gray(np.arange(256))))

    model_order = [
        "Classical_image_seg",
        "Rotation_invariant_image_seg",
        "Classical_image_SDF",
        "Rotation_invariant_image_SDF",
        "Rotation_invariant_pointcloud_SDF",
    ]

    for split, split_ids, gt_segs, gt_sdfs, gt_test_i_sdfs in [
        ("test", test_ids, gt_test_segs, gt_test_sdfs, gt_test_sdfs_iae)
    ]:

        num_rows = len(split_ids)
        num_columns = len(model_order) + 4
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(num_columns * 5, num_rows * 5))

        for i, c in enumerate(split_ids):
            gt_seg = gt_segs[i]
            gt_sdf = np.clip(gt_sdfs[i], -2, 2)
            gt_sdf_i = gt_test_i_sdfs[i].reshape(
                eval_scaled_img_resolution, eval_scaled_img_resolution, eval_scaled_img_resolution
            )
            row_index = i
            recons = []
            for m in model_order:
                recon_path = reconstructions_path + f"{m}/{c}.npy"
                recon = np.load(recon_path).squeeze()

                if "SDF" in m or "vn" in m:
                    mid_z = recon.shape[0] // 2
                    if ("Rotation" in m) and (m != "Rotation_invariant_pointcloud_SDF"):
                        z_slice = recon[mid_z, :, :].T
                        from scipy.ndimage import rotate

                        z_slice = rotate(z_slice, angle=-135, cval=2)
                        z_slice = z_slice[14:-14, 14:-14]
                    else:
                        z_slice = recon[:, :, mid_z].T
                else:
                    z_slice = recon.max(0)
                recons.append(z_slice)

            struc_seg = gt_orig_struc[i]
            cell_seg = gt_orig_cell[i]
            nuc_seg = gt_orig_nuc[i]

            axs[row_index, 0].imshow(
                cell_seg.max(0), cmap=cmap_inverted, origin="lower", alpha=0.5
            )
            axs[row_index, 0].imshow(nuc_seg.max(0), cmap=cmap_inverted, origin="lower", alpha=0.5)
            axs[row_index, 0].imshow(
                struc_seg.max(0), cmap=cmap_inverted, origin="lower", alpha=0.5
            )
            axs[row_index, 0].axis("off")
            axs[row_index, 0].set_title("")  # (f'GT Seg CellId {c}')

            axs[row_index, 1].imshow(gt_seg.max(0), cmap=cmap_inverted, origin="lower")
            axs[row_index, 1].axis("off")
            axs[row_index, 1].set_title("")  # (f'GT Seg CellId {c}')

            for i, img in enumerate(recons[:2]):
                axs[row_index, i + 2].imshow(img, cmap=cmap_inverted, origin="lower")
                axs[row_index, i + 2].axis("off")
                axs[row_index, i + 2].set_title("")  # run_to_displ_name[model_order[i]])

            axs[row_index, 4].imshow(
                gt_sdf[:, :, mid_slice_].T, cmap="seismic", origin="lower", vmin=-2, vmax=2
            )
            axs[row_index, 4].axis("off")
            axs[row_index, 4].set_title("")  # (f'GT SDF CellId {c}')

            for i, img in enumerate(recons[2:4]):
                axs[row_index, i + 5].imshow(img, cmap="seismic", origin="lower", vmin=-2, vmax=2)
                axs[row_index, i + 5].axis("off")
                axs[row_index, i + 5].set_title("")  # run_to_displ_name[model_order[i]])

            axs[row_index, 7].imshow(
                gt_sdf_i[:, :, mid_slice_].T.clip(-0.5, 0.5), cmap="seismic", origin="lower"
            )
            axs[row_index, 7].axis("off")
            axs[row_index, 7].set_title("")  # (f'GT SDF CellId {c}')

            axs[row_index, 8].imshow(recons[-1].clip(-0.5, 0.5), cmap="seismic", origin="lower")
            axs[row_index, 8].axis("off")
            axs[row_index, 8].set_title("")  # (f'GT SDF CellId {c}')

        plt.tight_layout()
        plt.savefig(reconstructions_path + "sample_recons.png", dpi=300, bbox_inches="tight")
        plt.savefig(reconstructions_path + "sample_recons.pdf", dpi=300, bbox_inches="tight")
