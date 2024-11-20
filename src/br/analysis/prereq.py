# Free up cache
import gc, torch
gc.collect()
torch.cuda.empty_cache()

import os, subprocess
import argparse
from pathlib import Path

# Based on the utilization, set the GPU ID

def get_gpu_info():
    # Run nvidia-smi command and get the output
    cmd = ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu", "--format=csv,noheader,nounits"]
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
            mig_id = line.split("(UUID: ")[-1].strip(')')
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from br.features.archetype import AA_Fast
from br.features.plot import collect_outputs, plot, plot_stratified_pc
from br.features.reconstruction import stratified_latent_walk
from br.features.utils import (
    normalize_intensities_and_get_colormap,
    normalize_intensities_and_get_colormap_apply,
)
from br.models.compute_features import compute_features, get_embeddings
from br.models.load_models import get_data_and_models
from br.models.save_embeddings import (
    get_pc_loss,
    get_pc_loss_chamfer,
    save_embeddings,
    save_emissions,
)
from br.models.utils import get_all_configs_per_dataset

def main(args):
    # Set working directory and paths
    os.chdir(args.src_path)
    save_path = args.save_path
    results_path = args.results_path
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    debug = args.debug

    # Load data and models
    data_list, all_models, run_names, model_sizes = get_data_and_models(
        dataset_name, batch_size, results_path, debug
    )

    # Save model sizes to CSV
    gg = pd.DataFrame()
    gg["model"] = run_names
    gg["model_size"] = model_sizes
    gg.to_csv(os.path.join(save_path, "model_sizes.csv"))

    compute_embeddings()
    compute_relevant_features()

def compute_embeddings():
    # Compute embeddings and reconstructions for each model
    debug = False
    splits_list = ["train", "val", "test"]
    meta_key = "rule"
    eval_scaled_img = [False] * 5
    eval_scaled_img_params = [{}] * 5
    loss_eval_list = None
    # sample_points_list = [True, True, False, False, False] # This is also different for each of PCNA and Cellpack - RITVIK
    sample_points_list = [False, False, True, True, False]
    skew_scale = 100

    save_embeddings(
        save_path,
        data_list,
        all_models,
        run_names,
        debug,
        splits_list,
        device,
        meta_key,
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
    ] # Different again


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
    parser.add_argument("--src_path", type=str, required=True,
                        help="Path to the source directory.")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save the embeddings.")
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to the results directory.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for processing.")
    parser.add_argument("--debug", type=bool, default=True,
                        help="Enable debug mode.")

 
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