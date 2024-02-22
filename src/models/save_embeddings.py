import os
from hydra.utils import instantiate
import yaml
import torch
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from src.models.predict_model import process_batch_embeddings, process_batch
import logging


def get_pc_loss():
    return instantiate(
        yaml.safe_load(
            """    
    _aux: earthmovers
    _target_: cyto_dl.nn.losses.GeomLoss
    p: 1
    blur: 0.01
    """
        )
    )


def get_pc_loss_chamfer():
    return instantiate(
        yaml.safe_load(
            """    
            _target_: cyto_dl.nn.losses.ChamferLoss
            """
        )
    )


def process_dataloader(
    dataloader,
    model,
    loss_eval_pc,
    track_emissions,
    emissions_path,
    split,
    all_embeds,
    all_data_ids,
    all_splits,
    all_loss,
    debug,
    device,
):
    for j, i in enumerate(tqdm(dataloader)):
        if (debug) and j > 1:
            break
        (
            all_embeds,
            all_data_ids,
            all_splits,
            all_loss,
        ) = process_batch_embeddings(
            model,
            loss_eval_pc,
            device,
            i,
            all_splits,
            all_data_ids,
            all_embeds,
            all_loss,
            split,
            track_emissions,
            emissions_path,
        )
    return all_embeds, all_data_ids, all_splits, all_loss


def compute_embeddings(
    model,
    this_data,
    split_list,
    loss_eval_pc,
    track_emissions,
    emissions_path,
    all_embeds,
    all_data_ids,
    all_splits,
    all_loss,
    debug,
    device,
):
    if "train" in split_list:
        print("Processing train")
        all_embeds, all_data_ids, all_splits, all_loss = process_dataloader(
            this_data.train_dataloader(),
            model,
            loss_eval_pc,
            track_emissions,
            emissions_path,
            "train",
            all_embeds,
            all_data_ids,
            all_splits,
            all_loss,
            debug,
            device,
        )
    if "val" in split_list:
        print("Processing val")
        all_embeds, all_data_ids, all_splits, all_loss = process_dataloader(
            this_data.val_dataloader(),
            model,
            loss_eval_pc,
            track_emissions,
            emissions_path,
            "val",
            all_embeds,
            all_data_ids,
            all_splits,
            all_loss,
            debug,
            device,
        )
    if "test" in split_list:
        print("Processing test")
        all_embeds, all_data_ids, all_splits, all_loss = process_dataloader(
            this_data.test_dataloader(),
            model,
            loss_eval_pc,
            track_emissions,
            emissions_path,
            "test",
            all_embeds,
            all_data_ids,
            all_splits,
            all_loss,
            debug,
            device,
        )
    return all_embeds, all_data_ids, all_splits, all_loss


def save_embeddings(
    save_folder: str = "./embeddings/",
    data_list: list = [],
    all_models: list = [],
    run_names: list = [],
    debug: bool = False,
    split_list: list = ["train", "val", "test"],
    device: str = "cuda:0",
    loss_eval_list: list = None,
):
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    track_emissions = False
    emissions_path = Path("./")

    for j_ind, model in enumerate(all_models):
        model = model.eval()
        all_data_ids = []
        all_embeds = []
        all_loss = []
        all_splits = []
        this_data = data_list[j_ind]
        loss_eval = get_pc_loss() if loss_eval_list is None else loss_eval_list[j_ind]
        with torch.no_grad():
            all_embeds, all_data_ids, all_splits, all_loss = compute_embeddings(
                model,
                this_data,
                split_list,
                loss_eval,
                track_emissions,
                emissions_path,
                all_embeds,
                all_data_ids,
                all_splits,
                all_loss,
                debug,
                device,
            )

            all_splits = [x for xs in all_splits for x in xs]
            all_data_ids = [x for xs in all_data_ids for x in xs]
            all_loss = [x for xs in all_loss for x in xs]

            all_embeds = np.concatenate(all_embeds, axis=0)
            all_embeds2 = all_embeds
            if len(all_embeds.shape) > 2:
                all_embeds2 = all_embeds[:, 1:, :].mean(axis=1)

            tmp_df = pd.DataFrame()
            tmp_df["CellId"] = all_data_ids
            tmp_df[[f"mu_{i}" for i in range(all_embeds2.shape[1])]] = all_embeds2
            tmp_df["loss"] = all_loss
            tmp_df["split"] = all_splits

            this_run_name = run_names[j_ind]

            tmp_df.to_csv(Path(save_folder) / f"{this_run_name}.csv")

            if len(all_embeds.shape) > 2:
                return all_embeds, all_loss, all_data_ids, all_splits


def save_emissions(
    emissions_path: str = "./emissions/",
    data_list: list = [],
    all_models: list = [],
    run_names: list = [],
    max_batches: int = 5,
    debug: bool = False,
    device: str = "cuda:0",
    loss_eval_list: list = None,
):
    emissions_path = Path(emissions_path)
    emissions_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    loss_eval_pc = get_pc_loss()

    if debug:
        max_batches = 1

    all_model_emissions = []
    for j_ind, model in enumerate(all_models):
        model = model.eval()
        all_data_ids, all_data_inputs, all_outputs = [], [], []
        all_embeds, all_emissions, all_x_vis_list = [], [], []
        all_loss = []
        all_splits = []
        this_data = data_list[j_ind]
        loss_eval = get_pc_loss() if loss_eval_list is None else loss_eval_list[j_ind]
        with torch.no_grad():
            count = 0
            for i in tqdm(this_data.test_dataloader()):
                if count < max_batches:
                    track_emissions = True
                else:
                    break
                count += 1
                (
                    all_data_inputs,
                    all_outputs,
                    all_embeds,
                    all_data_ids,
                    all_splits,
                    all_loss,
                    all_emissions,
                    all_x_vis_list,
                ) = process_batch(
                    model,
                    loss_eval,
                    device,
                    i,
                    all_data_inputs,
                    all_splits,
                    all_data_ids,
                    all_outputs,
                    all_embeds,
                    all_emissions,
                    all_x_vis_list,
                    all_loss,
                    "test",
                    track_emissions,
                    emissions_path,
                )
            all_model_emissions.append(pd.concat(all_emissions, axis=0))

    all_i = []
    for j_ind2, i in enumerate(all_model_emissions):
        i["model"] = run_names[j_ind2]
        all_i.append(i)
    emissions_df = pd.concat(all_i, axis=0).reset_index(drop=True)
    emissions_df.to_csv(Path(emissions_path) / "emissions.csv")
