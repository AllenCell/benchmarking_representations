import gc
import os
import subprocess

import torch

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
