# Free up cache
import argparse
import gc
import os
import subprocess

import pandas as pd
import torch

from br.models.compute_features import compute_features
from br.models.load_models import get_data_and_models
from br.models.save_embeddings import (
    get_pc_loss_chamfer,
    save_embeddings,
    save_emissions,
)
from br.models.utils import get_all_configs_per_dataset

gc.collect()
torch.cuda.empty_cache()

# Based on the utilization, set the GPU ID


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


selected_gpu_id_or_uuid = config_gpu()

# Set the CUDA_VISIBLE_DEVICES environment variable using the selected ID
if selected_gpu_id_or_uuid:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu_id_or_uuid
    print(f"CUDA_VISIBLE_DEVICES set to: {selected_gpu_id_or_uuid}")
else:
    print("No suitable GPU or MIG ID found. Exiting...")

# Set the device
device = "cuda:0"

# Setting a GPU ID is crucial for the script to work well!


def main(args):
    # Set working directory and paths
    os.chdir(args.src_path)
    save_path = args.save_path
    results_path = args.results_path
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    debug = args.debug

    # Load data and models
    data_list, all_models, run_names, model_sizes, manifest = get_data_and_models(
        dataset_name, batch_size, results_path, debug
    )

    # Save model sizes to CSV
    sizes_ = pd.DataFrame()
    sizes_["model"] = run_names
    sizes_["model_size"] = model_sizes
    sizes_.to_csv(os.path.join(save_path, "model_sizes.csv"))

    save_embeddings_across_models(args, manifest, data_list, all_models, run_names)
    compute_relevant_features()


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


def save_embeddings_across_models(args, manifest, data_list, all_models, run_names):
    """
    Save embeddings across models
    """
    # Compute embeddings and reconstructions for each model
    splits_list = ["train", "val", "test"]
    (
        eval_scaled_img,
        eval_scaled_img_params,
        loss_eval_list,
        sample_points_list,
        skew_scale,
    ) = _setup_evaluation_params(manifest, run_names)

    save_embeddings(
        args.save_path,
        data_list,
        all_models,
        run_names,
        args.debug,
        splits_list,
        device,
        args.meta_key,
        loss_eval_list,
        sample_points_list,
        skew_scale,
        eval_scaled_img,
        eval_scaled_img_params,
    )


def compute_relevant_features():

    batch_size = 1
    data_list, all_models, run_names, model_sizes = get_data_and_models(
        dataset_name, batch_size, results_path, debug
    )

    # Save emission stats for each model
    max_batches = 40
    save_emissions(
        save_path,
        data_list,
        all_models,
        run_names,
        max_batches,
        debug,
        device,
        loss_eval_list,
        sample_points_list,
        skew_scale,
        eval_scaled_img,
        eval_scaled_img_params,
    )

    # Compute multi-metric benchmarking features
    keys = ["pcloud"] * 5
    max_embed_dim = 256
    DATA_LIST = get_all_configs_per_dataset(results_path)
    data_config_list = DATA_LIST[dataset_name]["data_paths"]

    evolve_params = {
        "modality_list_evolve": keys,
        "config_list_evolve": data_config_list,
        "num_evolve_samples": 40,
        "compute_evolve_dataloaders": False,
        "eval_meshed_img": [False] * 5,
        "skew_scale": 100,
        "eval_meshed_img_model_type": [None] * 5,
        "only_embedding": False,
        "fit_pca": False,
    }

    loss_eval = get_pc_loss_chamfer()
    loss_eval_list = [loss_eval] * 5
    # use_sample_points_list = [True, True, False, False, False] # This again is different . RITVIK
    use_sample_points_list = [False, False, True, True, False]

    classification_params = {"class_labels": ["rule"]}
    rot_inv_params = {"squeeze_2d": False, "id": "cell_id", "max_batches": 4000}
    regression_params = {"df_feat": None, "target_cols": None, "feature_df_path": None}
    compactness_params = {
        "method": "mle",
        "num_PCs": None,
        "blobby_outlier_max_cc": None,
        "check_duplicates": True,
    }

    splits_list = ["train", "val", "test"]
    compute_embeds = False

    metric_list = [
        "Rotation Invariance Error",
        "Evolution Energy",
        "Reconstruction",
        "Classification",
        "Compactness",
    ]  # Different again

    compute_features(
        dataset=dataset_name,
        results_path=results_path,
        embeddings_path=save_path,
        save_folder=save_path,
        data_list=data_list,
        all_models=all_models,
        run_names=run_names,
        use_sample_points_list=use_sample_points_list,
        keys=keys,
        device=device,
        max_embed_dim=max_embed_dim,
        splits_list=splits_list,
        compute_embeds=compute_embeds,
        classification_params=classification_params,
        regression_params=regression_params,
        metric_list=metric_list,
        loss_eval_list=loss_eval_list,
        evolve_params=evolve_params,
        rot_inv_params=rot_inv_params,
        compactness_params=compactness_params,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for Benchmarking Representations")
    parser.add_argument(
        "--src_path", type=str, required=True, help="Path to the source directory."
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the embeddings."
    )
    parser.add_argument(
        "--results_path", type=str, required=True, help="Path to the results directory."
    )
    parser.add_argument(
        "--meta_key",
        type=str,
        required=True,
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
    if not args.src_path or not args.save_path or not args.results_path or not args.dataset_name:
        print("Error: Required arguments are missing.")
        sys.exit(1)

    main(args)

"""
Example
os.chdir(r"/allen/aics/assay-dev/users/Fatwir/benchmarking_representations/src/")
save_path = r"/allen/aics/assay-dev/users/Fatwir/benchmarking_representations/src/test_cellpack_save_embeddings/"
results_path = r"/allen/aics/assay-dev/users/Fatwir/benchmarking_representations/configs/results/"
dataset_name = "cellpack"
batch_size = 2
debug = True

"""
